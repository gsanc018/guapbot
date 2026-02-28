"""
guapbot/cli/portfolio_commands.py

Portfolio trading commands — runs both strategies simultaneously.

Workflow:
    1. guapbot features build XBTUSD     ← build feature cache (XBTUSD)
    2. guapbot features build ETHUSD     ← build feature cache (ETHUSD)
    3. guapbot portfolio run             ← run dual-strategy paper trading

Command structure:
    guapbot portfolio run
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

portfolio_app = typer.Typer(name="portfolio", help="Run dual-strategy portfolio simulation.")

_DEFAULT_TRAIN_END = "2024-06-30"
_DEFAULT_START     = "2025-01-01"

# Log-return column search order (mirrors paper_commands.py)
_LOG_RETURN_COL       = "1h_log_return"
_LOG_RETURN_FALLBACKS = ["log_return", "1h_log_ret_1", "log_ret_1"]

# Shared model registry (same as paper_commands.py)
_ML_MODEL_KEYS = {
    "gradient_boost": (".pkl", "guapbot.models.gradient_boost", "GradientBoost"),
    "lstm":           (".pt",  "guapbot.models.lstm",            "LSTMModel"),
    "rl_agent":       (".zip", "guapbot.models.rl_agent",        "RLAgent"),
}

_PRINT_EVERY = 24


# ---------------------------------------------------------------------------
# guapbot portfolio run
# ---------------------------------------------------------------------------

@portfolio_app.command("run")
def run(
    capital: float = typer.Option(
        10_000.0,
        "--capital",
        help="Total starting capital in USD (split between both strategies).",
    ),
    mp_fraction: float = typer.Option(
        0.60,
        "--mp-fraction",
        help="Baseline fraction of capital allocated to money_printer (default 0.60).",
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
        help="Last date of training split used to fit ensembles.",
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
        help="Per-strategy AND portfolio daily drawdown kill switch (default -5%).",
    ),
    total_dd_limit: float = typer.Option(
        -0.15,
        "--total-dd-limit",
        help="Per-strategy AND portfolio total drawdown kill switch (default -15%).",
    ),
    use_redis: bool = typer.Option(
        False,
        "--redis / --no-redis",
        help="Write state to Redis MarketState (requires Redis running).",
    ),
) -> None:
    """
    Run dual-strategy portfolio paper trading (money_printer + sat_stacker).

    Loads XBTUSD features for money_printer and ETHUSD features for
    sat_stacker, aligns them on a shared timestamp index, then replays
    both through the full execution pipeline simultaneously. The ETHBTC
    cross-signal dynamically shifts capital allocation between the two.
    """
    console.print("\n[bold]GuapBot Portfolio Paper Trading[/bold]")
    console.print(f"  Capital: {capital:,.2f} USD  |  mp={mp_fraction:.0%}  ss={1-mp_fraction:.0%}\n")

    # ------------------------------------------------------------------
    # Step 1: Load features for both pairs
    # ------------------------------------------------------------------
    try:
        from guapbot.features.pipeline import FeaturePipeline
    except ImportError as exc:
        console.print(f"[red]Import error: {exc}[/red]")
        raise typer.Exit(1)

    pipe = FeaturePipeline()
    console.print("  Loading XBTUSD features (money_printer)...")
    try:
        xbt_df = pipe.transform("XBTUSD")
    except FileNotFoundError:
        console.print("[red]No feature cache for XBTUSD. Run: guapbot features build XBTUSD[/red]")
        raise typer.Exit(1)

    console.print("  Loading ETHUSD features (sat_stacker)...")
    try:
        eth_df = pipe.transform("ETHUSD")
    except FileNotFoundError:
        console.print("[red]No feature cache for ETHUSD. Run: guapbot features build ETHUSD[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Step 2: Align on shared timestamp index
    # ------------------------------------------------------------------
    shared_idx = xbt_df.index.intersection(eth_df.index)
    if len(shared_idx) == 0:
        console.print("[red]No overlapping timestamps between XBTUSD and ETHUSD features.[/red]")
        raise typer.Exit(1)

    xbt_df = xbt_df.loc[shared_idx]
    eth_df = eth_df.loc[shared_idx]
    console.print(f"  Aligned: {len(shared_idx):,} shared bars")

    # ------------------------------------------------------------------
    # Step 3: Resolve log-return columns
    # ------------------------------------------------------------------
    xbt_lr_col = _resolve_log_return_col(xbt_df)
    eth_lr_col = _resolve_log_return_col(eth_df)
    if xbt_lr_col is None or eth_lr_col is None:
        console.print("[red]Missing log-return column in one or both feature caches.[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Step 4: Build + fit models for each pair
    # ------------------------------------------------------------------
    from guapbot.models.trend_following import TrendFollowing
    from guapbot.models.mean_reversion import MeanReversion

    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    xbt_train = xbt_df[xbt_df.index <= train_end_ts]
    eth_train = eth_df[eth_df.index <= train_end_ts]

    console.print("\n  [bold]money_printer[/bold] (XBTUSD) models:")
    mp_models = _build_models("XBTUSD", "money_printer", save_dir, xbt_train)

    console.print("\n  [bold]sat_stacker[/bold] (ETHUSD) models:")
    ss_models = _build_models("ETHUSD", "sat_stacker", save_dir, eth_train)

    # ------------------------------------------------------------------
    # Step 5: Fit ensembles on training data
    # ------------------------------------------------------------------
    from guapbot.cli.paper_commands import _build_signal_history
    mp_ensemble = _fit_ensemble("XBTUSD", mp_models, xbt_train, console)
    ss_ensemble = _fit_ensemble("ETHUSD", ss_models, eth_train, console)

    # ------------------------------------------------------------------
    # Step 6: Slice test data
    # ------------------------------------------------------------------
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end, tz="UTC") if end else xbt_df.index[-1]

    xbt_test = xbt_df[(xbt_df.index >= start_ts) & (xbt_df.index <= end_ts)]
    xbt_test = xbt_test.drop(columns=["target"], errors="ignore")
    eth_test = eth_df[(eth_df.index >= start_ts) & (eth_df.index <= end_ts)]
    eth_test = eth_test.drop(columns=["target"], errors="ignore")

    # Re-align test slices
    test_idx = xbt_test.index.intersection(eth_test.index)
    xbt_test = xbt_test.loc[test_idx]
    eth_test = eth_test.loc[test_idx]

    if xbt_test.empty:
        console.print(f"[red]No test data in [{start}, {end or 'end'}]. Adjust --start/--end.[/red]")
        raise typer.Exit(1)

    console.print(
        f"\n  Portfolio backtest: {len(xbt_test):,} bars "
        f"({xbt_test.index[0].date()} → {xbt_test.index[-1].date()})\n"
    )

    # ------------------------------------------------------------------
    # Step 7: Build portfolio components
    # ------------------------------------------------------------------
    mp_capital = capital * mp_fraction
    ss_capital = capital * (1.0 - mp_fraction)

    state = None
    if use_redis:
        from guapbot.execution.market_state import MarketState
        state = MarketState()
        state.connect()
        if not state.connected:
            console.print("[yellow]⚠[/yellow] Redis not available — running without MarketState")
            state = None
        else:
            console.print("  [green]✓[/green] MarketState connected to Redis")

    from guapbot.execution.paper_trader import PaperTrader
    from guapbot.portfolio import (
        PortfolioRunner,
        CrossSignals,
        CorrelationTracker,
        CapitalAllocator,
        BaseCurrencyAccounting,
        Benchmark,
        PortfolioRiskManager,
    )

    mp_trader = PaperTrader(
        models=mp_models,
        ensemble=mp_ensemble,
        initial_capital=mp_capital,
        fee_rate=fee_rate,
        max_long=0.25,
        max_short=0.15,
        daily_dd_limit=daily_dd_limit,
        total_dd_limit=total_dd_limit,
        state=state,
        pair="XBTUSD",
    )
    ss_trader = PaperTrader(
        models=ss_models,
        ensemble=ss_ensemble,
        initial_capital=ss_capital,
        fee_rate=fee_rate,
        max_long=0.20,
        max_short=0.0,    # long-only: accumulate sats
        daily_dd_limit=daily_dd_limit,
        total_dd_limit=total_dd_limit,
        state=state,
        pair="ETHUSD",
    )

    # Initial prices for benchmark (first bar close)
    initial_btc = _resolve_price(xbt_test.iloc[0].to_dict())
    initial_eth = _resolve_price(eth_test.iloc[0].to_dict())

    # Load portfolio config (YAML + CLI overrides)
    port_cfg = _load_portfolio_config(override_mp_fraction=mp_fraction)

    runner = PortfolioRunner(
        mp_trader=mp_trader,
        ss_trader=ss_trader,
        cross=CrossSignals(),
        correlation=CorrelationTracker(window=port_cfg["correlation_window"]),
        allocator=CapitalAllocator(
            base_split=port_cfg["base_split"],
            min_split=port_cfg["min_split"],
            max_split=port_cfg["max_split"],
            ethbtc_sensitivity=port_cfg["ethbtc_sensitivity"],
        ),
        accounting=BaseCurrencyAccounting(mp_capital, ss_capital, initial_btc),
        risk=PortfolioRiskManager(
            daily_dd_limit=daily_dd_limit,
            total_dd_limit=total_dd_limit,
        ),
        benchmark=Benchmark(initial_btc, initial_eth, capital),
    )

    # ------------------------------------------------------------------
    # Step 8: Run bar-by-bar replay
    # ------------------------------------------------------------------
    xbt_returns = xbt_test[xbt_lr_col].fillna(0.0).to_numpy(dtype=float)
    eth_returns = eth_test[eth_lr_col].fillna(0.0).to_numpy(dtype=float)
    n_bars = len(xbt_test)

    table = _make_table()
    kill_reason = ""

    for t in range(n_bars):
        mp_obs = xbt_test.iloc[t].to_dict()
        ss_obs = eth_test.iloc[t].to_dict()
        mp_lr  = float(xbt_returns[min(t + 1, n_bars - 1)])
        ss_lr  = float(eth_returns[min(t + 1, n_bars - 1)])

        result = runner.step(mp_obs, mp_lr, ss_obs, ss_lr)

        if t % _PRINT_EVERY == 0 or t == n_bars - 1 or result.kill_switch:
            _add_table_row(table, xbt_test.index[t], result)

        if result.kill_switch:
            kill_reason = result.kill_switch
            break

    console.print(table)

    # ------------------------------------------------------------------
    # Step 9: Summary
    # ------------------------------------------------------------------
    _print_summary(runner, capital, kill_reason)

    if state is not None:
        state.disconnect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_portfolio_config(override_mp_fraction: Optional[float] = None) -> dict:
    """
    Load portfolio.yaml from configs/ directory with fallback defaults.

    CLI flags override the YAML values (CLI flag > YAML > defaults).
    Returns a dict with keys: base_split, min_split, max_split,
    ethbtc_sensitivity, correlation_window.
    """
    defaults = {
        "base_split": 0.60,
        "min_split": 0.30,
        "max_split": 0.80,
        "ethbtc_sensitivity": 0.20,
        "correlation_window": 720,
    }

    config_path = Path("configs/portfolio.yaml")
    if config_path.exists():
        try:
            import yaml
            loaded = yaml.safe_load(config_path.read_text()) or {}
            for key in defaults:
                if key in loaded:
                    defaults[key] = loaded[key]
            logger.info("Loaded portfolio config from %s", config_path)
        except Exception as exc:
            logger.warning("Could not read %s: %s — using defaults", config_path, exc)

    # CLI --mp-fraction overrides base_split if explicitly provided
    if override_mp_fraction is not None:
        defaults["base_split"] = override_mp_fraction
        # Widen the min/max around the user-specified split
        defaults["min_split"] = max(0.10, override_mp_fraction - 0.30)
        defaults["max_split"] = min(0.90, override_mp_fraction + 0.30)

    return defaults


def _resolve_log_return_col(df: pd.DataFrame) -> Optional[str]:
    for col in [_LOG_RETURN_COL] + _LOG_RETURN_FALLBACKS:
        if col in df.columns:
            return col
    return None


def _resolve_price(obs: dict) -> float:
    for col in ["close", "price", "mid"]:
        val = obs.get(col, 0.0)
        if val and float(val) > 0:
            return float(val)
    return 1.0


def _build_models(pair: str, strategy: str, save_dir: Path, train_df: pd.DataFrame) -> list:
    from guapbot.models.trend_following import TrendFollowing
    from guapbot.models.mean_reversion import MeanReversion

    models = []
    for cls in [TrendFollowing, MeanReversion]:
        m = cls(pair, strategy)
        try:
            if not getattr(m, "_fitted", False):
                m.fit(train_df)
        except Exception as exc:
            logger.warning("Failed to fit %s for %s: %s", cls.__name__, pair, exc)
        models.append(m)

    for model_key, (ext, module_path, class_name) in _ML_MODEL_KEYS.items():
        model_file = save_dir / strategy / model_key / pair / f"model{ext}"
        if not model_file.exists():
            console.print(f"  [dim]Skipping {class_name} ({pair}) — not found[/dim]")
            continue
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls    = getattr(module, class_name)
            m      = cls(pair, strategy)
            m.load(str(model_file))
            models.append(m)
            console.print(f"  [green]✓[/green] {class_name} ({pair}) loaded")
        except Exception as exc:
            console.print(f"  [yellow]⚠[/yellow] Could not load {class_name} ({pair}): {exc}")

    console.print(f"  Total models ({pair}): {len(models)}")
    return models


def _fit_ensemble(pair: str, models: list, train_df: pd.DataFrame, con) -> Optional[object]:
    from guapbot.cli.paper_commands import _build_signal_history
    if train_df.empty:
        return None
    try:
        from guapbot.models.ensemble_lightgbm import EnsembleLearner
        signal_history = _build_signal_history(models, train_df)
        ens = EnsembleLearner(pair)
        ens.fit(signal_history)
        con.print(f"  [green]✓[/green] Ensemble fitted for {pair}")
        return ens
    except Exception as exc:
        con.print(f"  [yellow]⚠[/yellow] Ensemble fit failed for {pair}: {exc} — using signal average")
        return None


def _make_table() -> Table:
    table = Table(title="Portfolio Paper Trading — Bar Log", show_lines=False)
    table.add_column("Time",       style="dim",    width=17)
    table.add_column("MP Signal",  justify="right", width=9)
    table.add_column("MP Equity",  justify="right", width=11)
    table.add_column("SS Signal",  justify="right", width=9)
    table.add_column("SS Equity",  justify="right", width=11)
    table.add_column("Total USD",  justify="right", width=12)
    table.add_column("Sats",       justify="right", width=12)
    table.add_column("MP Frac",    justify="right", width=8)
    return table


def _add_table_row(table: Table, ts, result) -> None:
    snap = result.snapshot
    alloc = result.allocation
    ss_sats = f"{snap.ss_sats_accumulated:,}"
    table.add_row(
        str(ts)[:16],
        f"{result.mp_stats.signal:+.3f}",
        f"{snap.mp_equity_usd:,.2f}",
        f"{result.ss_stats.signal:+.3f}",
        f"{snap.ss_equity_usd:,.2f}",
        f"{snap.total_equity_usd:,.2f}",
        ss_sats,
        f"{alloc.money_printer_fraction:.0%}",
    )


def _print_summary(runner, initial_capital: float, kill_reason: str) -> None:
    from guapbot.execution.paper_trader import PaperTrader

    mp_eq  = runner.mp_equity
    ss_eq  = runner.ss_equity
    total  = runner.total_equity
    ret    = (total / initial_capital) - 1.0
    colour = "green" if ret >= 0 else "red"

    console.print("\n[bold]── Portfolio Summary ──[/bold]")
    console.print(f"  Bars processed   : {runner.bars_processed:,}")
    console.print(f"  money_printer    : {mp_eq:,.2f} USD")
    console.print(f"  sat_stacker      : {ss_eq:,.2f} USD")
    console.print(f"  Total equity     : {total:,.2f} USD")
    console.print(f"  Total return     : [{colour}]{ret * 100:+.2f}%[/{colour}]")
    if kill_reason:
        console.print(f"\n  [red bold]{kill_reason}[/red bold]")
    console.print()
