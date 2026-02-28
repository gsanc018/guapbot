"""
guapbot/data/aggregator.py

Tick data aggregation: converts raw Kraken trade rows into OHLCV bars.

Used by:
  - DataManager.backfill_trades()  — REST backfill (batch, one-time or quarterly)
  - BarBuilder.on_trade()          — live WebSocket stream (Session 7+)

The core function aggregate_trades_to_ohlcv() is shared by both paths so the
aggregation logic is never duplicated.
"""
from __future__ import annotations

import pandas as pd

from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# Map GuapBot interval strings → pandas resample frequency codes
INTERVAL_FREQ: dict[str, str] = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def rows_to_tick_df(rows: list) -> pd.DataFrame:
    """
    Parse raw Kraken trade rows into a typed tick DataFrame.

    Kraken GET /public/Trades row format (each row is a list):
      [price, volume, time, buy_sell, market_limit, misc, trade_id]
      - price:     string → float
      - volume:    string → float
      - time:      float  (Unix seconds with fractional part, e.g. 1735689600.123)
      - buy_sell:  "b" or "s"
      - market_limit: "m" or "l"
      - misc:      string (usually empty)
      - trade_id:  int

    Returns:
        DataFrame with columns:
          price     float64
          volume    float64
          timestamp datetime64[ns, UTC]
    """
    df = pd.DataFrame(
        rows,
        columns=["price", "volume", "time", "side", "type", "misc", "trade_id"],
    )
    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["price", "volume", "timestamp"]].reset_index(drop=True)


def aggregate_trades_to_ohlcv(ticks: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Aggregate a tick DataFrame into OHLCV bars.

    Args:
        ticks:    DataFrame with columns:
                    price     (float)
                    volume    (float)
                    timestamp (UTC-aware datetime)
        interval: '1h', '4h', or '1d'.

    Returns:
        OHLCV DataFrame indexed by UTC-aware timestamp (bar open time) with columns:
          open, high, low, close  float64
          volume                  float64
          trades                  int64
        Empty bars (periods with no trades) are dropped.
        Returns an empty DataFrame with the correct schema if ticks is empty.
    """
    if interval not in INTERVAL_FREQ:
        raise ValueError(f"Unknown interval '{interval}'. Use: {list(INTERVAL_FREQ)}")

    if ticks.empty:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "trades"]
        ).rename_axis("timestamp")

    freq = INTERVAL_FREQ[interval]
    ts_indexed = ticks.set_index("timestamp").sort_index()

    ohlcv = ts_indexed.resample(freq).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
        trades=("price", "count"),
    )

    # Drop empty bars (no trades during that period)
    ohlcv = ohlcv.dropna(subset=["open"])
    ohlcv["trades"] = ohlcv["trades"].astype(int)
    ohlcv.index.name = "timestamp"

    if not ohlcv.empty:
        logger.debug(
            "Aggregated %d ticks → %d %s bars (%s → %s)",
            len(ticks),
            len(ohlcv),
            interval,
            ohlcv.index[0].date(),
            ohlcv.index[-1].date(),
        )

    return ohlcv
