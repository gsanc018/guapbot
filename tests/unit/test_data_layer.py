"""
tests/unit/test_data_layer.py

Unit tests for Session 2: Data Layer.
All tests mock the Kraken API — no real network calls.

Run with: pytest tests/unit/test_data_layer.py -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from guapbot.data.aggregator import aggregate_trades_to_ohlcv, rows_to_tick_df
from guapbot.data.kraken_client import KrakenClient, KrakenRESTError
from guapbot.data.store import ParquetStore
from guapbot.data.manager import DataManager


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_ohlc_df(n: int = 10, start: str = "2024-01-01") -> pd.DataFrame:
    """Create a synthetic OHLC DataFrame for testing."""
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [50000.0 + i for i in range(n)],
            "high": [50100.0 + i for i in range(n)],
            "low":  [49900.0 + i for i in range(n)],
            "close":[50050.0 + i for i in range(n)],
            "volume":[1.0 + i * 0.1 for i in range(n)],
            "trades": [100 + i for i in range(n)],
        },
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


@pytest.fixture
def tmp_store(tmp_path: Path) -> ParquetStore:
    """ParquetStore backed by a temp directory."""
    return ParquetStore(cache_dir=tmp_path)


@pytest.fixture
def mock_client() -> MagicMock:
    """KrakenClient with mocked public methods."""
    client = MagicMock(spec=KrakenClient)
    client.get_ticker.return_value = {"ask": 50100.0, "bid": 49900.0, "last": 50000.0, "volume_24h": 1234.0}
    return client


@pytest.fixture
def manager(mock_client: MagicMock, tmp_store: ParquetStore) -> DataManager:
    return DataManager(client=mock_client, store=tmp_store)


# ------------------------------------------------------------------
# KrakenClient tests (mocked HTTP)
# ------------------------------------------------------------------

class TestKrakenClient:
    def test_get_ticker_parses_response(self) -> None:
        client = KrakenClient()
        fake_response = {
            "result": {
                "XXBTZUSD": {
                    "a": ["50100", 1, "50100.000"],
                    "b": ["49900", 1, "49900.000"],
                    "c": ["50000", "0.5"],
                    "v": ["100", "200"],
                }
            },
            "error": [],
        }
        with patch.object(client._session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = fake_response
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            ticker = client.get_ticker("XBTUSD")

        assert ticker["ask"] == 50100.0
        assert ticker["bid"] == 49900.0
        assert ticker["last"] == 50000.0


# ------------------------------------------------------------------
# ParquetStore tests
# ------------------------------------------------------------------

class TestParquetStore:
    def test_load_empty_when_no_cache(self, tmp_store: ParquetStore) -> None:
        df = tmp_store.load("XBTUSD", "1h")
        assert df.empty
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_save_and_load_roundtrip(self, tmp_store: ParquetStore) -> None:
        original = make_ohlc_df(100)
        tmp_store.save("XBTUSD", "1h", original)
        loaded = tmp_store.load("XBTUSD", "1h")
        assert len(loaded) == 100
        assert isinstance(loaded.index, pd.DatetimeIndex) and loaded.index.tz is not None

    def test_append_adds_new_bars(self, tmp_store: ParquetStore) -> None:
        first_batch = make_ohlc_df(10, start="2024-01-01")        # 00:00 – 09:00
        tmp_store.save("XBTUSD", "1h", first_batch)

        second_batch = make_ohlc_df(5, start="2024-01-01 10:00")  # 10:00 – 14:00 (no overlap)
        merged = tmp_store.append("XBTUSD", "1h", second_batch)

        assert len(merged) == 15  # 10 + 5, no overlap

    def test_append_deduplicates_on_overlap(self, tmp_store: ParquetStore) -> None:
        df1 = make_ohlc_df(10)
        df2 = make_ohlc_df(10)  # exact duplicate
        tmp_store.save("XBTUSD", "1h", df1)
        merged = tmp_store.append("XBTUSD", "1h", df2)
        assert len(merged) == 10

    def test_last_timestamp_none_when_empty(self, tmp_store: ParquetStore) -> None:
        assert tmp_store.last_timestamp("XBTUSD", "1h") is None

    def test_last_timestamp_returns_last_index(self, tmp_store: ParquetStore) -> None:
        df = make_ohlc_df(5)
        tmp_store.save("XBTUSD", "1h", df)
        ts = tmp_store.last_timestamp("XBTUSD", "1h")
        assert ts == df.index[-1]

    def test_cache_info_returns_metadata(self, tmp_store: ParquetStore) -> None:
        tmp_store.save("XBTUSD", "1h", make_ohlc_df(50))
        tmp_store.save("ETHUSD", "4h", make_ohlc_df(20))
        info = tmp_store.cache_info()
        assert "XBTUSD_1h" in info
        assert info["XBTUSD_1h"]["bars"] == 50
        assert "ETHUSD_4h" in info

    def test_clear_deletes_file(self, tmp_store: ParquetStore) -> None:
        tmp_store.save("XBTUSD", "1h", make_ohlc_df(5))
        assert tmp_store.last_timestamp("XBTUSD", "1h") is not None
        tmp_store.clear("XBTUSD", "1h")
        assert tmp_store.last_timestamp("XBTUSD", "1h") is None

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_zero_volume_bars_dropped(self, tmp_store: ParquetStore) -> None:
        df = make_ohlc_df(10).copy()
        df["volume"] = df["volume"].where(~df.index.isin(df.index[2:4]), 0.0)
        tmp_store.save("XBTUSD", "1h", df)
        loaded = tmp_store.load("XBTUSD", "1h")
        assert len(loaded) == 8
        assert (loaded["volume"] > 0).all()

    def test_negative_volume_bars_dropped(self, tmp_store: ParquetStore) -> None:
        df = make_ohlc_df(10).copy()
        df["volume"] = df["volume"].where(df.index != df.index[0], -1.0)
        tmp_store.save("XBTUSD", "1h", df)
        loaded = tmp_store.load("XBTUSD", "1h")
        assert len(loaded) == 9

    def test_ohlc_high_below_low_dropped(self, tmp_store: ParquetStore) -> None:
        df = make_ohlc_df(10).copy()
        # Swap high and low on bar 5 so high < low
        df["high"] = df["high"].where(df.index != df.index[5], 49000.0)
        df["low"] = df["low"].where(df.index != df.index[5], 51000.0)
        tmp_store.save("XBTUSD", "1h", df)
        loaded = tmp_store.load("XBTUSD", "1h")
        assert len(loaded) == 9

    def test_negative_price_dropped(self, tmp_store: ParquetStore) -> None:
        df = make_ohlc_df(10).copy()
        df["close"] = df["close"].where(df.index != df.index[0], -1.0)
        tmp_store.save("XBTUSD", "1h", df)
        loaded = tmp_store.load("XBTUSD", "1h")
        assert len(loaded) == 9

    def test_nan_ohlcv_dropped(self, tmp_store: ParquetStore) -> None:
        df = make_ohlc_df(10).copy()
        df["close"] = df["close"].where(df.index != df.index[3], float("nan"))
        tmp_store.save("XBTUSD", "1h", df)
        loaded = tmp_store.load("XBTUSD", "1h")
        assert len(loaded) == 9
        assert loaded["close"].notna().all()

    def test_duplicate_timestamps_deduplicated(self, tmp_store: ParquetStore) -> None:
        df = make_ohlc_df(10)
        df_with_dupe = pd.concat([df, df.iloc[[3]]])   # row 3 duplicated
        tmp_store.save("XBTUSD", "1h", df_with_dupe)
        loaded = tmp_store.load("XBTUSD", "1h")
        assert len(loaded) == 10
        assert loaded.index.is_unique

    def test_non_monotonic_index_sorted(self, tmp_store: ParquetStore) -> None:
        df = make_ohlc_df(10)
        df_shuffled = df.sample(frac=1, random_state=42)   # random order
        tmp_store.save("XBTUSD", "1h", df_shuffled)
        loaded = tmp_store.load("XBTUSD", "1h")
        assert loaded.index.is_monotonic_increasing

    def test_valid_data_passes_through_unchanged(self, tmp_store: ParquetStore) -> None:
        df = make_ohlc_df(20)
        tmp_store.save("XBTUSD", "1h", df)
        loaded = tmp_store.load("XBTUSD", "1h")
        assert len(loaded) == 20


# ------------------------------------------------------------------
# DataManager tests
# ------------------------------------------------------------------

class TestDataManager:
    def test_fetch_returns_empty_when_no_cache(
        self, manager: DataManager
    ) -> None:
        """fetch() with no cached data returns an empty DataFrame (no live API call)."""
        df = manager.fetch("XBTUSD", "1h")
        assert df.empty

    def test_fetch_returns_cached_data(
        self, manager: DataManager, tmp_store: ParquetStore
    ) -> None:
        """fetch() returns whatever is in the cache without making API calls."""
        existing = make_ohlc_df(100, start="2024-01-01")
        tmp_store.save("XBTUSD", "1h", existing)

        df = manager.fetch("XBTUSD", "1h")
        assert len(df) == 100

    def test_latest_price_calls_ticker(
        self, manager: DataManager, mock_client: MagicMock
    ) -> None:
        price = manager.latest_price("XBTUSD")
        assert price == 50000.0
        mock_client.get_ticker.assert_called_once_with("XBTUSD")

    def test_invalid_pair_raises(self, manager: DataManager) -> None:
        with pytest.raises(ValueError, match="Unknown pair"):
            manager.fetch("DOGEUSD", "1h")

    def test_invalid_interval_raises(self, manager: DataManager) -> None:
        with pytest.raises(ValueError, match="Unknown interval"):
            manager.fetch("XBTUSD", "5m")

    def test_available_pairs(self, manager: DataManager) -> None:
        pairs = manager.available_pairs()
        assert "XBTUSD" in pairs
        assert "ETHUSD" in pairs
        assert "ETHBTC" in pairs

    def test_cache_info_delegates_to_store(
        self, manager: DataManager, tmp_store: ParquetStore
    ) -> None:
        tmp_store.save("XBTUSD", "1h", make_ohlc_df(10))
        info = manager.cache_info()
        assert "XBTUSD_1h" in info

    def test_backfill_trades_paginates_and_saves(
        self, manager: DataManager, mock_client: MagicMock, tmp_store: ParquetStore
    ) -> None:
        """backfill_trades() paginates until cursor stalls, aggregates, and saves."""
        # Build synthetic tick rows: [price, volume, time, side, type, misc, trade_id]
        import time as time_mod
        base_ts = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
        # 120 trades spread across 2 hours so we get ≥2 1h bars
        tick_rows = [
            [str(50000 + i), str(0.01), base_ts + i * 60, "b", "m", "", i]
            for i in range(120)
        ]
        # First call returns 120 rows, cursor advances; second call stalls
        call_count = 0
        def get_trades_side_effect(pair, since=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tick_rows, base_ts + 120 * 60  # cursor advanced
            # Stall: last == since
            stall_cursor = base_ts + 120 * 60
            return [], stall_cursor

        mock_client.get_trades.side_effect = get_trades_side_effect

        results = manager.backfill_trades(pair="XBTUSD", since=0)

        assert "1h" in results
        assert len(results["1h"]) >= 1
        assert call_count == 2  # paginated exactly twice

    def test_backfill_trades_empty_response_returns_empty(
        self, manager: DataManager, mock_client: MagicMock
    ) -> None:
        """backfill_trades() returns {} when the API returns no trades."""
        mock_client.get_trades.return_value = ([], 0)
        results = manager.backfill_trades(pair="XBTUSD", since=0)
        assert results == {}

    def test_backfill_trades_unknown_pair_raises(self, manager: DataManager) -> None:
        with pytest.raises(ValueError, match="Unknown pair"):
            manager.backfill_trades(pair="DOGEUSD")


# ------------------------------------------------------------------
# Aggregator tests
# ------------------------------------------------------------------

def make_tick_rows(n: int, base_ts: int, price: float = 50000.0) -> list:
    """Build synthetic Kraken raw trade rows."""
    return [
        [str(price + i), str(0.01), float(base_ts + i * 60), "b", "m", "", i]
        for i in range(n)
    ]


class TestAggregator:
    def test_rows_to_tick_df_parses_kraken_format(self) -> None:
        base_ts = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
        rows = make_tick_rows(5, base_ts)
        df = rows_to_tick_df(rows)

        assert list(df.columns) == ["price", "volume", "timestamp"]
        assert df["price"].dtype == float
        assert df["volume"].dtype == float
        assert df["timestamp"].dt.tz is not None  # UTC-aware
        assert len(df) == 5

    def test_aggregate_1h_basic(self) -> None:
        """120 trades spanning 2 hours → 2 1h bars."""
        base_ts = int(pd.Timestamp("2026-01-01 00:00", tz="UTC").timestamp())
        rows = make_tick_rows(120, base_ts)
        ticks = rows_to_tick_df(rows)
        bars = aggregate_trades_to_ohlcv(ticks, "1h")

        assert len(bars) == 2
        assert list(bars.columns) == ["open", "high", "low", "close", "volume", "trades"]
        assert isinstance(bars.index, pd.DatetimeIndex)
        assert bars.index.tz is not None
        assert bars["trades"].dtype == int

    def test_aggregate_4h_groups_bars(self) -> None:
        """240 trades spanning 4 hours → 1 4h bar."""
        base_ts = int(pd.Timestamp("2026-01-01 00:00", tz="UTC").timestamp())
        rows = make_tick_rows(240, base_ts)
        ticks = rows_to_tick_df(rows)
        bars = aggregate_trades_to_ohlcv(ticks, "4h")

        assert len(bars) == 1
        assert bars.iloc[0]["volume"] == pytest.approx(240 * 0.01)
        assert bars.iloc[0]["trades"] == 240

    def test_aggregate_ohlc_values_correct(self) -> None:
        """open/high/low/close should match first/max/min/last trade in the period."""
        base_ts = int(pd.Timestamp("2026-01-01 00:00", tz="UTC").timestamp())
        rows = [
            ["50000", "1.0", float(base_ts + 0), "b", "m", "", 0],
            ["50500", "1.0", float(base_ts + 100), "b", "m", "", 1],
            ["49800", "1.0", float(base_ts + 200), "b", "m", "", 2],
            ["50200", "1.0", float(base_ts + 300), "b", "m", "", 3],
        ]
        ticks = rows_to_tick_df(rows)
        bars = aggregate_trades_to_ohlcv(ticks, "1h")

        assert len(bars) == 1
        bar = bars.iloc[0]
        assert bar["open"] == pytest.approx(50000.0)
        assert bar["high"] == pytest.approx(50500.0)
        assert bar["low"] == pytest.approx(49800.0)
        assert bar["close"] == pytest.approx(50200.0)

    def test_aggregate_empty_returns_empty(self) -> None:
        ticks = rows_to_tick_df([])
        bars = aggregate_trades_to_ohlcv(ticks, "1h")
        assert bars.empty
        assert list(bars.columns) == ["open", "high", "low", "close", "volume", "trades"]

    def test_aggregate_invalid_interval_raises(self) -> None:
        base_ts = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
        ticks = rows_to_tick_df(make_tick_rows(5, base_ts))
        with pytest.raises(ValueError, match="Unknown interval"):
            aggregate_trades_to_ohlcv(ticks, "5m")


# ------------------------------------------------------------------
# KrakenClient.get_trades tests
# ------------------------------------------------------------------

class TestKrakenClientTrades:
    def test_get_trades_parses_response(self) -> None:
        """get_trades() should return (list_of_rows, int_cursor)."""
        client = KrakenClient()
        base_ts = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
        fake_rows = [
            ["50000.00", "0.5", float(base_ts + i * 60), "b", "m", "", i]
            for i in range(5)
        ]
        fake_last = str(base_ts * 1_000_000_000 + 9999)
        fake_response = {
            "result": {"XXBTZUSD": fake_rows, "last": fake_last},
            "error": [],
        }

        with patch.object(client._session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = fake_response
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            rows, last_cursor = client.get_trades("XBTUSD", since=0)

        assert isinstance(rows, list)
        assert len(rows) == 5
        assert isinstance(last_cursor, int)
        assert last_cursor == int(fake_last)

    def test_get_trades_api_error_raises(self) -> None:
        client = KrakenClient()
        with patch.object(client._session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"result": {}, "error": ["EGeneral:Invalid arguments"]}
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            with pytest.raises(KrakenRESTError):
                client.get_trades("XBTUSD")