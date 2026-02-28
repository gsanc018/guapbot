"""
guapbot/cli/model_commands.py

Model training and inspection commands.

Workflow:
    1. guapbot features build XBTUSD        ← build normalized feature cache first
    2. guapbot train trend XBTUSD           ← instant (rule-based, no training)
    3. guapbot train xgb XBTUSD             ← ~5 minutes
    4. guapbot train lstm XBTUSD            ← ~30-60 minutes
    5. guapbot train rl XBTUSD              ← ~2-4 hours
    6. guapbot models info                  ← see what's saved

Model types:
    trend    — TrendFollowing (rule-based, instant)
    meanrev  — MeanReversion (rule-based, instant)
    xgb      — GradientBoost (XGBoost + LightGBM, ~5 min)
    lstm     — LSTMModel (PyTorch, ~30-60 min)
    rl       — RLAgent (Stable-Baselines3 SAC, ~2-4h)

Training data split (honoring handoff spec):
    Train:      Jan 2019 – train_end    (default: 2024-06-30)
    Validation: train_end+1 – val_end  (default: 2024-12-31)
    Test:       val_end+1 – present    (never touched during training)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from guapbot.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

model_app = typer.Typer(name="models", help="Model training and inspection.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_TYPES = {
    "trend": "TrendFollowing",
    "meanrev": "MeanReversion",
    "xgb": "GradientBoost",
    "lstm": "LSTMModel",
    "rl": "RLAgent",
}

_DEFAULT_TRAIN_END = "2024-06-30"
_DEFAULT_VAL_END = "2024-12-31"

# Column used to construct supervised targets from raw features
_RAW_LOG_RETURN_COL = "log_return"
_RAW_LOG_RETURN_FALLBACKS = ["log_ret_1", "1h_log_return"]


# ---------------------------------------------------------------------------
# guapbot train
# ---------------------------------------------------------------------------

@model_app.command("train")
def train(
    model_type: str = typer.Argument(
        ...,
        help="Model type: trend | meanrev | xgb | lstm | rl",
    ),
    pair: str = typer.Argument(..., help="Pair: XBTUSD or ETHUSD"),
    strategy: str = typer.Option(
        "money_printer",
        "--strategy", "-s",
        help="Strategy: money_printer or sat_stacker",
    ),
    algo: str = typer.Option(
        "sac",
        "--algo",
        help="RL algorithm (rl model only): sac | ppo | td3",
    ),
    train_end: str = typer.Option(
        _DEFAULT_TRAIN_END,
        "--train-end",
        help="Last date of training split (YYYY-MM-DD)",
    ),
    val_end: str = typer.Option(
        _DEFAULT_VAL_END,
        "--val-end",
        help="Last date of validation split (YYYY-MM-DD)",
    ),
    save_dir: Path = typer.Option(
        Path("models"),
        "--save-dir",
        help="Root directory for saved models",
    ),
) -> None:
    """
    Train a trading model and save it to disk.

    Loads the normalized feature cache for PAIR, creates the target column
    from raw log returns, applies the train/val date split, then fits and
    saves the chosen model.
    """
    model_type = model_type.lower()
    pair = pair.upper()

    if model_type not in _MODEL_TYPES:
        console.print(
            f"[red]Unknown model type '{model_type}'. "
            f"Choose from: {', '.join(_MODEL_TYPES)}[/red]"
        )
        raise typer.Exit(1)

    if strategy not in ("money_printer", "sat_stacker"):
        console.print("[red]strategy must be money_printer or sat_stacker[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold]GuapBot Train[/bold] — {_MODEL_TYPES[model_type]} "
        f"for {pair} ({strategy})"
    )

    # ------------------------------------------------------------------
    # Step 1: Load features
    # ------------------------------------------------------------------
    try:
        from guapbot.features.pipeline import FeaturePipeline
    except ImportError as exc:
        console.print(f"[red]Failed to import FeaturePipeline: {exc}[/red]")
        raise typer.Exit(1)

    pipe = FeaturePipeline()

    console.print("  Loading normalized features...")
    try:
        norm_df = pipe.transform(pair)
    except FileNotFoundError:
        console.print(
            f"[red]No feature cache for {pair}. "
            f"Run: guapbot features build {pair}[/red]"
        )
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Step 2: Load raw features and build target (supervised models only)
    # ------------------------------------------------------------------
    is_supervised = model_type in ("xgb", "lstm")
    target_series: Optional[pd.Series] = None

    if is_supervised:
        raw_path = pipe._raw_path(pair)
        if raw_path.exists():
            console.print("  Building target from raw log returns...")
            raw_df = pd.read_parquet(raw_path)
            raw_df = raw_df.reindex(norm_df.index).sort_index()
            log_ret_col = _resolve_log_return_col(raw_df)
            if log_ret_col:
                # Next-bar direction: shift back by 1 (future return)
                target_series = np.sign(raw_df[log_ret_col].shift(-1))
                target_series.name = "target"
            else:
                console.print(
                    "[yellow]No log-return column in raw features — "
                    "using sign of normalized log_return as target proxy[/yellow]"
                )
                log_ret_col = _resolve_log_return_col(norm_df)
                if log_ret_col:
                    target_series = np.sign(norm_df[log_ret_col].shift(-1))
                    target_series.name = "target"
        else:
            console.print(
                "[yellow]Raw feature cache not found — "
                "using normalized log_return sign as target[/yellow]"
            )
            log_ret_col = _resolve_log_return_col(norm_df)
            if log_ret_col:
                target_series = np.sign(norm_df[log_ret_col].shift(-1))
                target_series.name = "target"

        if target_series is None:
            console.print(
                "[red]Could not construct target column — "
                "no log-return column found.[/red]"
            )
            raise typer.Exit(1)

        # Inject target and drop last row (NaN target)
        norm_df = norm_df.copy()
        norm_df["target"] = target_series
        norm_df = norm_df.dropna(subset=["target"])
        norm_df = norm_df[norm_df["target"] != 0]  # drop flat bars

    # ------------------------------------------------------------------
    # Step 3: Apply train/val date split
    # ------------------------------------------------------------------
    try:
        train_end_ts = pd.Timestamp(train_end, tz="UTC")
        val_end_ts = pd.Timestamp(val_end, tz="UTC")
    except Exception:
        console.print(
            f"[red]Invalid date format. Use YYYY-MM-DD, got "
            f"train_end={train_end!r}, val_end={val_end!r}[/red]"
        )
        raise typer.Exit(1)

    train_df = norm_df[norm_df.index <= train_end_ts]
    val_df = norm_df[(norm_df.index > train_end_ts) & (norm_df.index <= val_end_ts)]

    if train_df.empty:
        console.print(
            f"[red]No training data before {train_end}. "
            "Run guapbot features build first.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"  Train: {len(train_df):,} bars "
        f"({train_df.index[0].date()} → {train_df.index[-1].date()})"
    )
    if not val_df.empty:
        console.print(
            f"  Val:   {len(val_df):,} bars "
            f"({val_df.index[0].date()} → {val_df.index[-1].date()})"
        )

    # ------------------------------------------------------------------
    # Step 4: Instantiate and train model
    # ------------------------------------------------------------------
    model = _make_model(model_type, pair, strategy, algo)
    console.print(f"  Training {_MODEL_TYPES[model_type]}...")

    try:
        model.fit(train_df)
    except Exception as exc:
        console.print(f"[red]Training failed: {exc}[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Step 5: Validation report (supervised only)
    # ------------------------------------------------------------------
    if is_supervised and not val_df.empty:
        _print_val_report(model, val_df, model_type)

    # ------------------------------------------------------------------
    # Step 6: Save model
    # ------------------------------------------------------------------
    model_dir = save_dir / strategy / model_type / pair
    model_dir.mkdir(parents=True, exist_ok=True)

    ext = _save_extension(model_type)
    save_path = str(model_dir / f"model{ext}")

    try:
        model.save(save_path)
        console.print(f"\n[bold green]Saved:[/bold green] {save_path}")
    except NotImplementedError:
        # Rule-based models have no weights
        console.print(
            f"\n[bold green]Done.[/bold green] "
            f"{_MODEL_TYPES[model_type]} is rule-based — no file to save.\n"
            "Re-create and fit from features at runtime."
        )
    except Exception as exc:
        console.print(f"[red]Save failed: {exc}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# guapbot models info
# ---------------------------------------------------------------------------

@model_app.command("info")
def info(
    save_dir: Path = typer.Option(Path("models"), "--save-dir"),
) -> None:
    """Show all saved models and their metadata."""
    if not save_dir.exists():
        console.print(f"[yellow]No models directory found at {save_dir}[/yellow]")
        console.print("Run [bold]guapbot train[/bold] to train your first model.")
        return

    table = Table(title="GuapBot Saved Models", header_style="bold cyan")
    table.add_column("Strategy")
    table.add_column("Model")
    table.add_column("Pair")
    table.add_column("File")
    table.add_column("Size")

    found = False
    for strategy_dir in sorted(save_dir.iterdir()):
        if not strategy_dir.is_dir():
            continue
        for model_dir in sorted(strategy_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for pair_dir in sorted(model_dir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                for f in sorted(pair_dir.iterdir()):
                    if f.suffix in (".pkl", ".pt", ".zip", ".npy"):
                        size_kb = round(f.stat().st_size / 1024, 1)
                        table.add_row(
                            strategy_dir.name,
                            model_dir.name,
                            pair_dir.name,
                            f.name,
                            f"{size_kb} KB",
                        )
                        found = True

    if found:
        console.print(table)
    else:
        console.print("[yellow]No saved model files found.[/yellow]")
        console.print("Run [bold]guapbot train[/bold] to train models.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_model(model_type: str, pair: str, strategy: str, algo: str):
    """Instantiate the correct model class."""
    if model_type == "trend":
        from guapbot.models.trend_following import TrendFollowing
        return TrendFollowing(pair, strategy)
    elif model_type == "meanrev":
        from guapbot.models.mean_reversion import MeanReversion
        return MeanReversion(pair, strategy)
    elif model_type == "xgb":
        from guapbot.models.gradient_boost import GradientBoost
        return GradientBoost(pair, strategy)
    elif model_type == "lstm":
        from guapbot.models.lstm import LSTMModel
        return LSTMModel(pair, strategy)
    elif model_type == "rl":
        from guapbot.models.rl_agent import RLAgent
        return RLAgent(pair, strategy, algo=algo)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


def _save_extension(model_type: str) -> str:
    """Return the file extension for the model type."""
    return {
        "xgb": ".pkl",
        "lstm": ".pt",
        "rl": "",   # SB3 adds .zip automatically; save path is bare
    }.get(model_type, "")


def _resolve_log_return_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first available log-return column in df."""
    for col in [_RAW_LOG_RETURN_COL] + _RAW_LOG_RETURN_FALLBACKS:
        if col in df.columns:
            return col
    return None


def _print_val_report(model, val_df: pd.DataFrame, model_type: str) -> None:
    """Print directional accuracy on the validation split."""
    try:
        signals = []
        targets = []
        for _, row in val_df.iterrows():
            obs = row.drop("target").to_dict()
            sig = model.predict(obs)
            tgt = float(row["target"])
            signals.append(np.sign(sig) if sig != 0 else 0)
            targets.append(tgt)

        if not signals:
            return

        correct = sum(1 for s, t in zip(signals, targets) if s == t and s != 0)
        non_flat = sum(1 for s in signals if s != 0)
        acc = correct / non_flat if non_flat > 0 else 0.0
        flat_pct = (len(signals) - non_flat) / len(signals) * 100

        console.print(
            f"  [cyan]Val accuracy:[/cyan] {acc:.1%} "
            f"({correct}/{non_flat} non-flat bars, {flat_pct:.0f}% flat)"
        )
    except Exception as exc:
        logger.debug("Val report failed: %s", exc)
