"""
guapbot/cli/backtest_commands.py

Backtest commands.

Workflow:
    1. guapbot features build XBTUSD       ← build feature cache first
    2. guapbot train trend XBTUSD           ← (optional) train ML models
    3. guapbot train xgb XBTUSD
    4. guapbot backtest run XBTUSD          ← run backtest, print metrics
    5. guapbot backtest run XBTUSD --out /tmp/bt.pkl
       guapbot backtest report --result /tmp/bt.pkl --out /tmp/report.html

Command structure:
    guapbot backtest run PAIR
    guapbot backtest report
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from guapbot.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

backtest_app = typer.Typer(name="backtest", help="Backtest trading strategies.")

_DEFAULT_TRAIN_END = "2024-06-30"
_DEFAULT_START     = "2025-01-01"

# Maps CLI model key → (class path, save extension)
_ML_MODEL_KEYS = {
    "gradient_boost": (".pkl", "guapbot.models.gradient_boost", "GradientBoost"),
    "lstm":           (".pt",  "guapbot.models.lstm",            "LSTMModel"),
    "rl_agent":       (".zip", "guapbot.models.rl_agent",        "RLAgent"),
}

_ROLLING_SHARPE_WINDOW = 720   # 30 days × 24 h


# ---------------------------------------------------------------------------
# guapbot backtest run
# ---------------------------------------------------------------------------

@backtest_app.command("run")
def run(
    pair: str = typer.Argument(..., help="Pair: XBTUSD or ETHUSD"),
    strategy: str = typer.Option(
        "money_printer",
        "--strategy", "-s",
        help="Strategy: money_printer or sat_stacker",
    ),
    start: str = typer.Option(
        _DEFAULT_START,
        "--start",
        help="Backtest start date (YYYY-MM-DD). Default: test split start.",
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="Backtest end date (YYYY-MM-DD). Default: end of cache.",
    ),
    train_end: str = typer.Option(
        _DEFAULT_TRAIN_END,
        "--train-end",
        help="Last date of training split for ensemble fitting.",
    ),
    save_dir: Path = typer.Option(
        Path("models"),
        "--save-dir",
        help="Root directory of saved models.",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Save BacktestResult to this .pkl path (for later 'report' use).",
    ),
) -> None:
    """
    Run a full backtest for PAIR and print a performance summary.

    Loads all available trained models plus always-on rule-based models.
    Builds an EnsembleLearner on the training split, then backtests on
    the date range [start, end].
    """
    pair = pair.upper()
    console.print(f"\n[bold]GuapBot Backtest[/bold] — {pair} ({strategy})")

    # ------------------------------------------------------------------
    # Step 1: Load normalized features
    # ------------------------------------------------------------------
    try:
        from guapbot.features.pipeline import FeaturePipeline
    except ImportError as exc:
        console.print(f"[red]Import error: {exc}[/red]")
        raise typer.Exit(1)

    pipe = FeaturePipeline()
    console.print("  Loading normalized features...")
    try:
        full_df = pipe.transform(pair)
    except FileNotFoundError:
        console.print(
            f"[red]No feature cache for {pair}. "
            f"Run: guapbot features build {pair}[/red]"
        )
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Step 2: Load / create models
    # ------------------------------------------------------------------
    from guapbot.models.trend_following import TrendFollowing
    from guapbot.models.mean_reversion import MeanReversion

    models = []
    trend   = TrendFollowing(pair, strategy)
    meanrev = MeanReversion(pair, strategy)

    # Fit rule-based models on any non-empty slice to set _fitted=True
    _fit_rule_based(trend,   full_df)
    _fit_rule_based(meanrev, full_df)
    models.extend([trend, meanrev])
    console.print("  [green]✓[/green] TrendFollowing + MeanReversion (rule-based)")

    # Try to load saved ML models
    for model_key, (ext, module_path, class_name) in _ML_MODEL_KEYS.items():
        model_file = save_dir / strategy / model_key / pair / f"model{ext}"
        if not model_file.exists():
            console.print(f"  [dim]Skipping {class_name} — not found at {model_file}[/dim]")
            continue
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls    = getattr(module, class_name)
            m      = cls(pair, strategy)
            m.load(str(model_file))
            models.append(m)
            console.print(f"  [green]✓[/green] {class_name} loaded from {model_file}")
        except Exception as exc:
            console.print(f"  [yellow]⚠[/yellow] Could not load {class_name}: {exc}")

    console.print(f"  Total models: {len(models)}")

    # ------------------------------------------------------------------
    # Step 3: Build signal_history and fit EnsembleLearner
    # ------------------------------------------------------------------
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    train_df     = full_df[full_df.index <= train_end_ts]

    if train_df.empty:
        console.print(
            f"[red]No training data before {train_end}. "
            "Run guapbot features build first.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"  Building signal_history on {len(train_df):,} training bars..."
    )
    signal_history = _build_signal_history(models, train_df)

    from guapbot.models.ensemble_lightgbm import EnsembleLearner
    ensemble = EnsembleLearner(pair)
    console.print("  Fitting EnsembleLearner (walk-forward stacking)...")
    try:
        ensemble.fit(signal_history)
        console.print("  [green]✓[/green] Ensemble fitted")
    except Exception as exc:
        console.print(f"  [yellow]⚠[/yellow] Ensemble fit failed: {exc} — using signal average")
        ensemble = None

    # ------------------------------------------------------------------
    # Step 4: Run backtest on test split
    # ------------------------------------------------------------------
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end, tz="UTC") if end else full_df.index[-1]

    test_df = full_df[(full_df.index >= start_ts) & (full_df.index <= end_ts)]
    if test_df.empty:
        console.print(
            f"[red]No data in [{start}, {end or 'end'}]. "
            "Adjust --start / --end.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"  Backtesting {len(test_df):,} bars "
        f"({test_df.index[0].date()} → {test_df.index[-1].date()})..."
    )

    from guapbot.backtest.engine import BacktestEngine
    engine = BacktestEngine(models=models, ensemble=ensemble)

    try:
        result = engine.run(test_df)
    except Exception as exc:
        console.print(f"[red]Backtest failed: {exc}[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Step 5: Print report
    # ------------------------------------------------------------------
    from guapbot.backtest.report import print_report
    print_report(result, title=f"GuapBot Backtest — {pair} {strategy}")

    # ------------------------------------------------------------------
    # Step 6: Optionally save result
    # ------------------------------------------------------------------
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(result, fh)
        console.print(f"[green]Result saved →[/green] {out}")
        console.print(
            f"  Render HTML: [bold]guapbot backtest report "
            f"--result {out} --out report.html[/bold]"
        )


# ---------------------------------------------------------------------------
# guapbot backtest report
# ---------------------------------------------------------------------------

@backtest_app.command("report")
def report(
    result_path: Path = typer.Option(
        ...,
        "--result",
        help="Path to a pickled BacktestResult (.pkl) from 'guapbot backtest run --out'.",
    ),
    out: Path = typer.Option(
        Path("report.html"),
        "--out",
        help="Output HTML file path.",
    ),
    title: str = typer.Option(
        "GuapBot Backtest",
        "--title",
        help="Report title shown in charts.",
    ),
) -> None:
    """
    Render a Plotly HTML report from a saved BacktestResult.

    Requires plotly: pip install plotly
    """
    if not result_path.exists():
        console.print(f"[red]Result file not found: {result_path}[/red]")
        raise typer.Exit(1)

    console.print(f"  Loading result from {result_path}...")
    with open(result_path, "rb") as fh:
        result = pickle.load(fh)

    from guapbot.backtest.report import save_html, print_report
    print_report(result, title=title)

    try:
        save_html(result, str(out), title=title)
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_rule_based(model, df: pd.DataFrame) -> None:
    """Fit a rule-based model on a small slice just to set _fitted=True."""
    try:
        model.fit(df.iloc[:min(10, len(df))])
    except Exception as exc:
        logger.debug("Rule-based fit failed: %s", exc)


def _build_signal_history(models, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all models over train_df bar-by-bar to produce signal_history.

    Returns a DataFrame with columns matching EnsembleLearner._FEATURE_COLS + 'target'.
    Rolling Sharpe computed as the 30-day rolling Sharpe of each model's PnL series.
    """
    from guapbot.models.ensemble_lightgbm import _MODEL_NAMES

    # Map from model class name → canonical model name
    _class_to_name = {
        "trendfollowing": "trend_following",
        "meanreversion":  "mean_reversion",
        "gradientboost":  "gradient_boost",
        "lstmmodel":      "lstm",
        "rlagent":        "rl_agent",
    }

    n = len(train_df)
    records: dict[str, list] = {m: [] for m in _MODEL_NAMES}
    confs:   dict[str, list] = {m: [] for m in _MODEL_NAMES}

    # Resolve log-return column for target and rolling Sharpe
    _lr_cols = ["1h_log_return", "log_return", "1h_log_ret_1", "log_ret_1"]
    lr_col = next((c for c in _lr_cols if c in train_df.columns), None)
    log_returns = train_df[lr_col].fillna(0.0).to_numpy() if lr_col else np.zeros(n)

    # Per-model PnL series (signal * next_bar_log_return) for rolling Sharpe
    pnl_series: dict[str, list] = {m: [] for m in _MODEL_NAMES}

    for t in range(n):
        obs = train_df.iloc[t].to_dict()
        next_lr = log_returns[min(t + 1, n - 1)]

        for model in models:
            cls_key = type(model).__name__.lower()
            name    = _class_to_name.get(cls_key, cls_key)
            if name not in _MODEL_NAMES:
                continue
            try:
                sig  = float(model.predict(obs))
                conf = float(model.confidence(obs))
            except Exception:
                sig, conf = 0.0, 0.0

            records[name].append(sig)
            confs[name].append(conf)
            pnl_series[name].append(sig * next_lr)

    # Pad any missing model columns with zeros
    for m in _MODEL_NAMES:
        if not records[m]:
            records[m] = [0.0] * n
            confs[m]   = [0.0] * n
            pnl_series[m] = [0.0] * n

    df_out = pd.DataFrame(index=train_df.index)

    # Compute rolling Sharpe per model
    for m in _MODEL_NAMES:
        pnl = pd.Series(pnl_series[m], index=train_df.index)
        roll_mean = pnl.rolling(_ROLLING_SHARPE_WINDOW, min_periods=1).mean()
        roll_std  = pnl.rolling(_ROLLING_SHARPE_WINDOW, min_periods=1).std().replace(0, np.nan)
        roll_sharpe = (roll_mean / roll_std * np.sqrt(8_760)).fillna(0.0)

        df_out[f"{m}_signal"]         = records[m]
        df_out[f"{m}_confidence"]     = confs[m]
        df_out[f"{m}_rolling_sharpe"] = roll_sharpe.values

    # Neutral regimes (no detectors in Session 6)
    df_out["regime_1h_label"]       = 0   # trending
    df_out["regime_1h_confidence"]  = 0.5
    df_out["regime_4h_label"]       = 0
    df_out["regime_4h_confidence"]  = 0.5
    df_out["regime_daily_label"]    = 0   # bullish
    df_out["regime_daily_confidence"] = 0.5

    # Target: next-bar direction
    if lr_col:
        next_log_ret = np.roll(log_returns, -1)
        next_log_ret[-1] = 0.0
        target = (next_log_ret > 0).astype(int)
        df_out["target"] = target
    else:
        df_out["target"] = 1  # degenerate but won't crash

    return df_out.dropna()
