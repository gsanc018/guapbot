"""
scripts/walk_forward.py

Walk-forward signal history generator.

Runs all available trained models (plus always-on rule-based models) bar-by-bar
over the full feature cache and saves the resulting signal_history parquet.
This parquet is the training input for EnsembleLearner.fit().

Usage:
    python scripts/walk_forward.py XBTUSD
    python scripts/walk_forward.py XBTUSD --save-dir models/ --out data/cache/
    python scripts/walk_forward.py ETHUSD --strategy sat_stacker

Output:
    data/cache/signal_history_XBTUSD.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_KEYS = {
    "gradient_boost": (".pkl", "guapbot.models.gradient_boost", "GradientBoost"),
    "lstm":           (".pt",  "guapbot.models.lstm",           "LSTMModel"),
    "rl_agent":       (".zip", "guapbot.models.rl_agent",       "RLAgent"),
}

_CLASS_TO_NAME = {
    "trendfollowing": "trend_following",
    "meanreversion":  "mean_reversion",
    "gradientboost":  "gradient_boost",
    "lstmmodel":      "lstm",
    "rlagent":        "rl_agent",
}

_MODEL_NAMES = ["trend_following", "mean_reversion", "gradient_boost", "lstm", "rl_agent"]
_ROLLING_WINDOW = 720  # 30 days × 24h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate signal_history parquet for EnsembleLearner training."
    )
    parser.add_argument("pair", help="Asset pair: XBTUSD or ETHUSD")
    parser.add_argument("--strategy", default="money_printer",
                        help="Strategy subdirectory in save-dir (default: money_printer)")
    parser.add_argument("--save-dir", default="models",
                        help="Root directory of saved models (default: models/)")
    parser.add_argument("--out", default="data/cache",
                        help="Output directory for signal_history parquet (default: data/cache/)")
    args = parser.parse_args()

    pair      = args.pair.upper()
    strategy  = args.strategy
    save_dir  = Path(args.save_dir)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Walk-forward signal history — {pair} ({strategy})")

    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    try:
        from guapbot.features.pipeline import FeaturePipeline
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    pipe = FeaturePipeline()
    print("  Loading normalized features...")
    try:
        full_df = pipe.transform(pair)
    except FileNotFoundError:
        print(f"ERROR: No feature cache for {pair}. Run: guapbot features build {pair}")
        sys.exit(1)

    print(f"  {len(full_df):,} bars loaded.")

    # ------------------------------------------------------------------
    # Load / create models
    # ------------------------------------------------------------------
    from guapbot.models.trend_following import TrendFollowing
    from guapbot.models.mean_reversion  import MeanReversion

    models = []
    trend   = TrendFollowing(pair, strategy)
    meanrev = MeanReversion(pair, strategy)
    for m in (trend, meanrev):
        try:
            m.fit(full_df.iloc[:10])
        except Exception:
            pass
    models.extend([trend, meanrev])
    print("  ✓ TrendFollowing + MeanReversion")

    for model_key, (ext, module_path, class_name) in _MODEL_KEYS.items():
        mfile = save_dir / strategy / model_key / pair / f"model{ext}"
        if not mfile.exists():
            print(f"  Skipping {class_name} (not found)")
            continue
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls    = getattr(module, class_name)
            m      = cls(pair, strategy)
            m.load(str(mfile))
            models.append(m)
            print(f"  ✓ {class_name}")
        except Exception as exc:
            print(f"  ⚠ Could not load {class_name}: {exc}")

    print(f"  Total models: {len(models)}")

    # ------------------------------------------------------------------
    # Generate signal history
    # ------------------------------------------------------------------
    print(f"  Generating signals for {len(full_df):,} bars...")

    _lr_cols = ["1h_log_return", "log_return", "1h_log_ret_1", "log_ret_1"]
    lr_col   = next((c for c in _lr_cols if c in full_df.columns), None)
    log_rets = full_df[lr_col].fillna(0.0).to_numpy() if lr_col else np.zeros(len(full_df))
    n        = len(full_df)

    records:    dict[str, list] = {m: [] for m in _MODEL_NAMES}
    confs:      dict[str, list] = {m: [] for m in _MODEL_NAMES}
    pnl_series: dict[str, list] = {m: [] for m in _MODEL_NAMES}

    for t in range(n):
        if t % 5000 == 0:
            print(f"    bar {t:,}/{n:,}")
        obs     = full_df.iloc[t].to_dict()
        next_lr = log_rets[min(t + 1, n - 1)]

        for model in models:
            name = _CLASS_TO_NAME.get(type(model).__name__.lower(), "")
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

    # Pad missing models
    for m in _MODEL_NAMES:
        while len(records[m]) < n:
            records[m].append(0.0)
            confs[m].append(0.0)
            pnl_series[m].append(0.0)

    # Build DataFrame
    df_out = pd.DataFrame(index=full_df.index)

    for m in _MODEL_NAMES:
        pnl = pd.Series(pnl_series[m], index=full_df.index)
        roll_mean   = pnl.rolling(_ROLLING_WINDOW, min_periods=1).mean()
        roll_std    = pnl.rolling(_ROLLING_WINDOW, min_periods=1).std().replace(0, np.nan)
        roll_sharpe = (roll_mean / roll_std * np.sqrt(8_760)).fillna(0.0)

        df_out[f"{m}_signal"]          = records[m]
        df_out[f"{m}_confidence"]      = confs[m]
        df_out[f"{m}_rolling_sharpe"]  = roll_sharpe.values

    df_out["regime_1h_label"]        = 0
    df_out["regime_1h_confidence"]   = 0.5
    df_out["regime_4h_label"]        = 0
    df_out["regime_4h_confidence"]   = 0.5
    df_out["regime_daily_label"]     = 0
    df_out["regime_daily_confidence"] = 0.5

    if lr_col:
        next_lr_arr = np.roll(log_rets, -1)
        next_lr_arr[-1] = 0.0
        df_out["target"] = (next_lr_arr > 0).astype(int)
    else:
        df_out["target"] = 1

    df_out = df_out.dropna()

    out_path = out_dir / f"signal_history_{pair}.parquet"
    df_out.to_parquet(out_path)
    print(f"\n  Saved → {out_path}  ({len(df_out):,} rows, {len(df_out.columns)} cols)")


if __name__ == "__main__":
    main()
