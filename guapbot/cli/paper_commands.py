"""
guapbot/cli/paper_commands.py

Paper trading commands.

Workflow:
    1. guapbot features build XBTUSD       ← build feature cache first
    2. guapbot train trend XBTUSD           ← (optional) train rule-based models
    3. guapbot train xgb XBTUSD             ← (optional) train ML models
    4. guapbot paper run XBTUSD            ← run paper trading, stream bar table

Command structure:
    guapbot paper run PAIR
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

paper_app = typer.Typer(name="paper", help="Run paper (simulated) trading.")

_DEFAULT_TRAIN_END = "2024-06-30"
_DEFAULT_START     = "2025-01-01"

# Maps model key → (file extension, module path, class name)
_ML_MODEL_KEYS = {
    "gradient_boost": (".pkl", "guapbot.models.gradient_boost", "GradientBoost"),
    "lstm":           (".pt",  "guapbot.models.lstm",            "LSTMModel"),
    "rl_agent":       (".zip", "guapbot.models.rl_agent",        "RLAgent"),
}

# Log-return column search order
_LOG_RETURN_COL       = "1h_log_return"
_LOG_RETURN_FALLBACKS = ["log_return", "1h_log_ret_1", "log_ret_1"]

# Print a live table row every N bars to avoid flooding the terminal
_PRINT_EVERY = 24


# ---------------------------------------------------------------------------
# guapbot paper run
# ---------------------------------------------------------------------------

@paper_app.command("run")
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
        help="Start date for paper trading (YYYY-MM-DD).",
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD). Default: end of feature cache.",
    ),
    train_end: str = typer.Option(
        _DEFAULT_TRAIN_END,
        "--train-end",
        help="Last date of training split used to fit the ensemble.",
    ),
    capital: float = typer.Option(
        10_000.0,
        "--capital",
        help="Starting capital in USD.",
    ),
    fee_rate: float = typer.Option(
        0.0026,
        "--fee-rate",
        help="Taker fee rate (default 0.26%).",
    ),
    save_dir: Path = typer.Option(
        Path("models"),
        "--save-dir",
        help="Root directory for saved models.",
    ),
    daily_dd_limit: float = typer.Option(
        -0.05,
        "--daily-dd-limit",
        help="Daily drawdown kill switch (default -5%).",
    ),
    total_dd_limit: float = typer.Option(
        -0.15,
        "--total-dd-limit",
        help="Total drawdown kill switch (default -15%).",
    ),
    use_redis: bool = typer.Option(
        False,
        "--redis / --no-redis",
        help="Write state to Redis MarketState (requires Redis running).",
    ),
) -> None:
    """
    Run paper trading for PAIR and stream a live table to the terminal.

    Loads all available trained models, fits an EnsembleLearner on the
    training split, then replays the feature cache from --start onwards
    through the full execution pipeline (position sizer → order executor).
    Kill switches halt the loop if drawdown limits are breached.
    """
    pair = pair.upper()
    console.print(f"\n[bold]GuapBot Paper Trading[/bold] — {pair} ({strategy})")

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
    # Step 2: Resolve log-return column
    # ------------------------------------------------------------------
    log_ret_col = _resolve_log_return_col(full_df)
    if log_ret_col is None:
        console.print(
            "[red]No log-return column found in feature cache. "
            f"Expected one of: {[_LOG_RETURN_COL] + _LOG_RETURN_FALLBACKS}[/red]"
        )
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Step 3: Load / create models
    # ------------------------------------------------------------------
    from guapbot.models.trend_following import TrendFollowing
    from guapbot.models.mean_reversion import MeanReversion

    models = []
    trend   = TrendFollowing(pair, strategy)
    meanrev = MeanReversion(pair, strategy)
    _fit_rule_based(trend,   full_df)
    _fit_rule_based(meanrev, full_df)
    models.extend([trend, meanrev])
    console.print("  [green]✓[/green] TrendFollowing + MeanReversion (rule-based)")

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
    # Step 4: Build signal_history and fit EnsembleLearner
    # ------------------------------------------------------------------
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    train_df     = full_df[full_df.index <= train_end_ts]

    ensemble = None
    if not train_df.empty:
        console.print(
            f"  Building signal_history on {len(train_df):,} training bars..."
        )
        signal_history = _build_signal_history(models, train_df)

        try:
            from guapbot.models.ensemble_lightgbm import EnsembleLearner
            ens = EnsembleLearner(pair)
            console.print("  Fitting EnsembleLearner (walk-forward stacking)...")
            ens.fit(signal_history)
            ensemble = ens
            console.print("  [green]✓[/green] Ensemble fitted")
        except Exception as exc:
            console.print(
                f"  [yellow]⚠[/yellow] Ensemble fit failed: {exc} — using signal average"
            )
    else:
        console.print(
            f"  [yellow]⚠[/yellow] No training data before {train_end} — skipping ensemble"
        )

    # ------------------------------------------------------------------
    # Step 5: Slice test data
    # ------------------------------------------------------------------
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end, tz="UTC") if end else full_df.index[-1]
    test_df  = full_df[(full_df.index >= start_ts) & (full_df.index <= end_ts)]

    # Drop target column if present — not a feature
    test_df = test_df.drop(columns=["target"], errors="ignore")

    if test_df.empty:
        console.print(
            f"[red]No data in [{start}, {end or 'end'}]. "
            "Adjust --start / --end.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"\n  Paper trading {len(test_df):,} bars "
        f"({test_df.index[0].date()} → {test_df.index[-1].date()})...\n"
    )

    # ------------------------------------------------------------------
    # Step 6: Initialise paper trader + optional MarketState
    # ------------------------------------------------------------------
    state = None
    if use_redis:
        from guapbot.execution.market_state import MarketState
        state = MarketState()
        state.connect()
        if not state.connected:
            console.print(
                "[yellow]⚠[/yellow] Redis not available — running without MarketState"
            )
            state = None
        else:
            console.print("  [green]✓[/green] MarketState connected to Redis")

    from guapbot.execution.paper_trader import PaperTrader

    trader = PaperTrader(
        models=models,
        ensemble=ensemble,
        initial_capital=capital,
        fee_rate=fee_rate,
        daily_dd_limit=daily_dd_limit,
        total_dd_limit=total_dd_limit,
        state=state,
        pair=pair,
    )

    # ------------------------------------------------------------------
    # Step 7: Run bar-by-bar replay with live Rich table
    # ------------------------------------------------------------------
    table = _make_table()
    log_returns = test_df[log_ret_col].fillna(0.0).to_numpy(dtype=float)
    n_bars      = len(test_df)

    kill_reason = ""
    n_printed   = 0

    for t in range(n_bars):
        obs        = test_df.iloc[t].to_dict()
        log_return = float(log_returns[min(t + 1, n_bars - 1)])

        stats = trader.step(obs, log_return)

        # Print a row every _PRINT_EVERY bars (and always the last bar)
        if t % _PRINT_EVERY == 0 or t == n_bars - 1 or stats.kill_switch:
            _add_table_row(table, test_df.index[t], stats)
            n_printed += 1

        if stats.kill_switch:
            kill_reason = stats.kill_switch
            break

    console.print(table)

    # ------------------------------------------------------------------
    # Step 8: Summary
    # ------------------------------------------------------------------
    result = trader.summary()
    _print_summary(result, capital, kill_reason)

    if state is not None:
        state.disconnect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_log_return_col(df: pd.DataFrame) -> Optional[str]:
    for col in [_LOG_RETURN_COL] + _LOG_RETURN_FALLBACKS:
        if col in df.columns:
            return col
    return None


def _fit_rule_based(model, df: pd.DataFrame) -> None:
    """Fit a rule-based model if it isn't already fitted."""
    try:
        if not getattr(model, "_fitted", False):
            model.fit(df)
    except Exception as exc:
        logger.warning("Failed to fit %s: %s", model, exc)


def _build_signal_history(models: list, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all models over train_df bar-by-bar to generate signal_history.

    Returns a DataFrame with per-model signal + confidence columns + target.
    Rolling Sharpe is approximated as the rolling mean PnL / std of PnL.
    """
    from guapbot.models.ensemble_lightgbm import _MODEL_NAMES

    _class_to_name = {
        "trendfollowing": "trend_following",
        "meanreversion":  "mean_reversion",
        "gradientboost":  "gradient_boost",
        "lstmmodel":      "lstm",
        "rlagent":        "rl_agent",
    }

    _ROLLING_SHARPE_WINDOW = 720  # 30d × 24h

    # Resolve log-return column
    log_ret_col = _resolve_log_return_col(train_df)
    log_returns = (
        train_df[log_ret_col].fillna(0.0).to_numpy(dtype=float)
        if log_ret_col else np.zeros(len(train_df))
    )

    feature_df = train_df.drop(columns=["target"], errors="ignore")
    n = len(feature_df)

    signals:   dict[str, list[float]] = {m: [] for m in _MODEL_NAMES}
    confs:     dict[str, list[float]] = {m: [] for m in _MODEL_NAMES}
    pnl_series: dict[str, list[float]] = {m: [] for m in _MODEL_NAMES}

    for t in range(n):
        obs = feature_df.iloc[t].to_dict()
        next_ret = float(log_returns[min(t + 1, n - 1)])

        for model in models:
            name = _class_to_name.get(type(model).__name__.lower(), "")
            if name not in _MODEL_NAMES:
                continue
            try:
                sig  = float(model.predict(obs))
                conf = float(model.confidence(obs))
            except Exception:
                sig, conf = 0.0, 0.0
            signals[name].append(sig)
            confs[name].append(conf)
            pnl_series[name].append(sig * next_ret)

        # Fill missing models with 0
        for name in _MODEL_NAMES:
            if len(signals[name]) < t + 1:
                signals[name].append(0.0)
                confs[name].append(0.0)
                pnl_series[name].append(0.0)

    records: dict[str, list] = {}
    for name in _MODEL_NAMES:
        records[f"{name}_signal"]     = signals[name]
        records[f"{name}_confidence"] = confs[name]
        pnl = pd.Series(pnl_series[name])
        roll_mean = pnl.rolling(_ROLLING_SHARPE_WINDOW, min_periods=1).mean()
        roll_std  = pnl.rolling(_ROLLING_SHARPE_WINDOW, min_periods=1).std().fillna(1e-8)
        records[f"{name}_rolling_sharpe"] = (roll_mean / roll_std).tolist()

    # Neutral regime features (no live regime detector in backtest)
    for tf in ("1h", "4h", "daily"):
        records[f"regime_{tf}_label"]      = [0.0] * n
        records[f"regime_{tf}_confidence"] = [0.5] * n

    # Binary target: 1 if next bar is up
    next_rets = np.roll(log_returns, -1)
    next_rets[-1] = 0.0
    records["target"] = (next_rets > 0).astype(float).tolist()

    return pd.DataFrame(records, index=feature_df.index)


def _make_table() -> Table:
    table = Table(title="Paper Trading — Bar Log", show_lines=False)
    table.add_column("Time",      style="dim",    width=20)
    table.add_column("Signal",    justify="right", width=8)
    table.add_column("Position",  justify="right", width=10)
    table.add_column("Log Ret",   justify="right", width=10)
    table.add_column("PnL %",     justify="right", width=8)
    table.add_column("Equity",    justify="right", width=12)
    return table


def _add_table_row(table: Table, ts, stats) -> None:
    from guapbot.execution.paper_trader import BarStats
    pnl_colour = "green" if stats.pnl_pct >= 0 else "red"
    table.add_row(
        str(ts)[:16],
        f"{stats.signal:+.3f}",
        f"{stats.position:+.4f}",
        f"{stats.log_return:+.4f}",
        f"[{pnl_colour}]{stats.pnl_pct * 100:+.3f}%[/{pnl_colour}]",
        f"{stats.equity:,.2f}",
    )


def _print_summary(result, initial_capital: float, kill_reason: str) -> None:
    ret = result.total_return
    ret_colour = "green" if ret >= 0 else "red"

    console.print("\n[bold]── Paper Trading Summary ──[/bold]")
    console.print(f"  Bars processed : {result.bars_processed:,}")
    console.print(f"  Trades         : {result.n_trades}")
    console.print(f"  Final equity   : {result.final_equity:,.2f}")
    console.print(
        f"  Total return   : [{ret_colour}]{ret * 100:+.2f}%[/{ret_colour}]"
    )
    if kill_reason:
        console.print(f"\n  [red bold]{kill_reason}[/red bold]")
    console.print()
