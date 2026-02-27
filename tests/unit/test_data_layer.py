"""
tests/unit/test_data_layer.py

Unit tests for Session 2: Data Layer.
All tests mock the Kraken API — no real network calls.

Run with: pytest tests/unit/test_data_layer.py -v
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
    """KrakenClient with mocked get_ohlc."""
    client = MagicMock(spec=KrakenClient)
    client.get_ohlc.return_value = make_ohlc_df(50)
    client.get_ticker.return_value = {"ask": 50100.0, "bid": 49900.0, "last": 50000.0, "volume_24h": 1234.0}
    return client


@pytest.fixture
def manager(mock_client: MagicMock, tmp_store: ParquetStore) -> DataManager:
    return DataManager(client=mock_client, store=tmp_store)


# ------------------------------------------------------------------
# KrakenClient tests (mocked HTTP)
# ------------------------------------------------------------------

class TestKrakenClient:
    def test_get_ohlc_parses_response(self) -> None:
        """KrakenClient.get_ohlc should return a properly shaped DataFrame."""
        client = KrakenClient()
        # Build a fake Kraken OHLC response
        now = int(time.time())
        fake_bars = [
            [now - 3600 * i, "50000", "50100", "49900", "50050", "50025", "1.5", 100]
            for i in range(5, 0, -1)
        ]
        fake_response = {
            "result": {
                "XXBTZUSD": fake_bars,
                "last": now,
            },
            "error": [],
        }

        with patch.object(client._session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = fake_response
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            df = client.get_ohlc("XBTUSD", "1h")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["open", "high", "low", "close", "volume", "trades"]
        assert df.index.tz is not None  # UTC-aware
        assert df.index.is_monotonic_increasing

    def test_get_ohlc_invalid_interval_raises(self) -> None:
        client = KrakenClient()
        with pytest.raises(ValueError, match="Unknown interval"):
            client.get_ohlc("XBTUSD", "5m")

    def test_get_ohlc_api_error_raises(self) -> None:
        client = KrakenClient()
        with patch.object(client._session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"result": {}, "error": ["EGeneral:Invalid arguments"]}
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            with pytest.raises(KrakenRESTError):
                client.get_ohlc("XBTUSD", "1h")

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
        assert loaded.index.tz is not None

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
# DataManager tests
# ------------------------------------------------------------------

class TestDataManager:
    def test_fetch_triggers_full_history_when_no_cache(
        self, manager: DataManager, mock_client: MagicMock
    ) -> None:
        """First fetch on empty cache should call get_ohlc multiple times (backfill)."""
        # Make client return empty eventually to stop the loop
        call_count = 0
        def side_effect(pair, interval, since=None):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                return pd.DataFrame()
            return make_ohlc_df(50, start="2024-01-01")

        mock_client.get_ohlc.side_effect = side_effect
        df = manager.fetch("XBTUSD", "1h")
        assert not df.empty

    def test_fetch_incremental_on_existing_cache(
        self, manager: DataManager, mock_client: MagicMock, tmp_store: ParquetStore
    ) -> None:
        """Second fetch should only request new bars since last cached timestamp."""
        # Pre-populate cache
        existing = make_ohlc_df(100, start="2024-01-01")
        tmp_store.save("XBTUSD", "1h", existing)

        # Fetch should now do incremental — single call with since set
        new_bars = make_ohlc_df(5, start="2024-01-05 04:00")
        mock_client.get_ohlc.return_value = new_bars

        df = manager.fetch("XBTUSD", "1h")
        call_args = mock_client.get_ohlc.call_args
        assert call_args.kwargs.get("since") is not None or call_args.args[2] is not None

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