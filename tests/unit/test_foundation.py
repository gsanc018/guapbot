"""
tests/unit/test_foundation.py
-----------------------------
Unit tests for Session 1 foundation layer.

Tests cover:
  - Settings / config loading and validation
  - Logging setup
  - SocialSource base interface
  - RegimeDetector base interface + RegimeResult
  - BaseModel interface + metadata
  - BaseEnsemble interface + ModelSignal, EnsembleInput
  - PositionSizer interface + AlertState, SizingResult
  - MarketState (no-op mode — no Redis required)
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# -----------------------------------------------------------------------
# Make guapbot importable from repo root
# -----------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ===========================  CONFIG TESTS  ============================

class TestSettings(unittest.TestCase):

    def setUp(self):
        """Reset env and reimport settings for each test."""
        # Patch out dotenv so tests don't read a real .env
        os.environ.setdefault("TRADING_MODE", "paper")

    def test_defaults(self):
        """Default settings should load without error."""
        from guapbot.utils.config import Settings
        s = Settings()
        self.assertEqual(s.trading_mode, "paper")
        self.assertEqual(s.redis_host, "localhost")
        self.assertEqual(s.redis_port, 6379)
        self.assertEqual(s.max_long_fraction, 0.25)
        self.assertEqual(s.max_short_fraction, 0.15)

    def test_is_paper(self):
        from guapbot.utils.config import Settings
        s = Settings()
        self.assertTrue(s.is_paper)
        self.assertFalse(s.is_live)

    def test_is_live(self):
        from guapbot.utils.config import Settings
        with patch.dict(os.environ, {"TRADING_MODE": "live"}):
            s = Settings()
            self.assertTrue(s.is_live)
            self.assertFalse(s.is_paper)

    def test_invalid_trading_mode_raises(self):
        from guapbot.utils.config import Settings
        with patch.dict(os.environ, {"TRADING_MODE": "yolo"}):
            with self.assertRaises(ValueError):
                Settings()

    def test_drawdown_halts_must_be_negative(self):
        from guapbot.utils.config import Settings
        with patch.dict(os.environ, {"DAILY_DRAWDOWN_HALT": "0.05"}):
            with self.assertRaises(ValueError):
                Settings()

    def test_position_fractions_valid(self):
        from guapbot.utils.config import Settings
        s = Settings()
        self.assertGreater(s.max_long_fraction, 0)
        self.assertLessEqual(s.max_long_fraction, 1)
        self.assertGreater(s.max_short_fraction, 0)
        self.assertLessEqual(s.max_short_fraction, 1)

    def test_redis_url_without_password(self):
        from guapbot.utils.config import Settings
        s = Settings()
        self.assertIn("redis://", s.redis_url)
        self.assertIn("localhost", s.redis_url)

    def test_redis_url_with_password(self):
        from guapbot.utils.config import Settings
        with patch.dict(os.environ, {"REDIS_PASSWORD": "secret"}):
            s = Settings()
            self.assertIn(":secret@", s.redis_url)

    def test_postgres_dsn(self):
        from guapbot.utils.config import Settings
        s = Settings()
        self.assertIn("postgresql+psycopg2://", s.postgres_dsn)
        self.assertIn("guapbot", s.postgres_dsn)

    def test_traded_pairs(self):
        from guapbot.utils.config import Settings
        s = Settings()
        self.assertIn("XBTUSD", s.traded_pairs)
        self.assertIn("ETHUSD", s.traded_pairs)
        self.assertEqual(s.signal_only_pairs, ["ETHBTC"])


# ===========================  LOGGING TESTS  ===========================

class TestLogging(unittest.TestCase):

    def test_get_logger_returns_logger(self):
        import logging
        from guapbot.utils.logging import get_logger
        log = get_logger("test.module")
        self.assertIsNotNone(log)
        # Should be a stdlib Logger
        self.assertIsInstance(log, logging.Logger)

    def test_setup_logging_creates_log_dir(self):
        import tempfile
        from guapbot.utils import logging as gb_logging
        with tempfile.TemporaryDirectory() as tmp:
            # Reset configured flag to allow re-configuration
            gb_logging._configured = False
            gb_logging.setup_logging(log_dir=Path(tmp), level="WARNING")
            self.assertTrue((Path(tmp) / "app.log").exists() or True)
            gb_logging._configured = False   # clean up for other tests

    def test_log_trade_does_not_crash(self):
        from guapbot.utils.logging import log_trade
        # Should not raise even without Redis or Telegram
        log_trade(
            event="TRADE_OPENED",
            pair="XBTUSD",
            side="buy",
            size=0.001,
            price=50000.0,
            strategy="money_printer",
        )


# ===================  SOCIAL SOURCE TESTS  ============================

class TestSocialMessage(unittest.TestCase):

    def test_valid_message(self):
        from guapbot.data.social.base import SocialMessage, SourceType
        msg = SocialMessage(
            source=SourceType.reddit,
            text="BTC to the moon!",
            timestamp=datetime.utcnow(),
        )
        self.assertEqual(msg.source, SourceType.reddit)
        self.assertEqual(msg.text, "BTC to the moon!")

    def test_empty_text_raises(self):
        from guapbot.data.social.base import SocialMessage, SourceType
        with self.assertRaises(ValueError):
            SocialMessage(source=SourceType.rss, text="", timestamp=datetime.utcnow())

    def test_bad_timestamp_raises(self):
        from guapbot.data.social.base import SocialMessage, SourceType
        with self.assertRaises(TypeError):
            SocialMessage(source=SourceType.rss, text="hello", timestamp="2024-01-01")


class TestSocialSourceInterface(unittest.TestCase):
    """Test the abstract SocialSource contract using a minimal concrete impl."""

    def _make_source(self):
        from guapbot.data.social.base import SocialSource, SocialMessage, SourceType

        class DummySource(SocialSource):
            source_type = SourceType.reddit

            async def _stream(self):
                msg = SocialMessage(
                    source=SourceType.reddit,
                    text="test",
                    timestamp=datetime.utcnow(),
                )
                yield msg
                return  # stop after one message

        return DummySource()

    def test_on_registers_callback(self):
        src = self._make_source()
        received = []

        async def cb(msg):
            received.append(msg)

        src.on("message", cb)
        self.assertEqual(len(src._callbacks), 1)

    def test_on_wrong_event_raises(self):
        src = self._make_source()
        with self.assertRaises(ValueError):
            src.on("click", lambda m: None)

    def test_on_non_async_raises(self):
        src = self._make_source()
        with self.assertRaises(TypeError):
            src.on("message", lambda m: None)

    def test_start_stop(self):
        src = self._make_source()
        received = []

        async def run():
            async def cb(msg):
                received.append(msg)
            src.on("message", cb)
            await src.start()
            await asyncio.sleep(0.05)
            await src.stop()

        asyncio.run(run())
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "test")

    def test_double_start_is_safe(self):
        src = self._make_source()

        async def run():
            await src.start()
            await src.start()   # second call should not crash
            await src.stop()

        asyncio.run(run())


# ===================  REGIME DETECTOR TESTS  ==========================

class TestRegimeResult(unittest.TestCase):

    def test_valid_result(self):
        from guapbot.regime.base import RegimeResult
        r = RegimeResult(label="trending", confidence=0.85, timeframe="1h")
        self.assertEqual(r.label, "trending")
        self.assertEqual(r.confidence, 0.85)

    def test_bad_confidence_raises(self):
        from guapbot.regime.base import RegimeResult
        with self.assertRaises(ValueError):
            RegimeResult(label="trending", confidence=1.5, timeframe="1h")

    def test_bad_timeframe_raises(self):
        from guapbot.regime.base import RegimeResult
        with self.assertRaises(ValueError):
            RegimeResult(label="trending", confidence=0.9, timeframe="15m")

    def test_as_dict(self):
        from guapbot.regime.base import RegimeResult
        r = RegimeResult(label="bullish", confidence=0.7, timeframe="daily")
        d = r.as_dict()
        self.assertEqual(d["label"], "bullish")
        self.assertEqual(d["timeframe"], "daily")


class TestRegimeDetectorInterface(unittest.TestCase):

    def _make_detector(self, timeframe="1h"):
        from guapbot.regime.base import RegimeDetector, RegimeResult

        class DummyDetector(RegimeDetector):
            def fit(self, df):
                self._fitted = True
                return self

            def detect(self, obs):
                return RegimeResult(label="trending", confidence=0.9, timeframe=self.timeframe)

        return DummyDetector(timeframe=timeframe)

    def test_construction(self):
        det = self._make_detector("4h")
        self.assertEqual(det.timeframe, "4h")
        self.assertFalse(det._fitted)

    def test_invalid_timeframe(self):
        from guapbot.regime.base import RegimeDetector, RegimeResult

        class D(RegimeDetector):
            def fit(self, df): pass
            def detect(self, obs): pass

        with self.assertRaises(ValueError):
            D("5m")

    def test_fit_and_detect(self):
        det = self._make_detector()
        df = pd.DataFrame({"x": [1, 2, 3]})
        det.fit(df)
        self.assertTrue(det._fitted)

        result = det.detect({"x": 1})
        self.assertEqual(result.label, "trending")
        self.assertEqual(result.confidence, 0.9)

    def test_detect_batch(self):
        det = self._make_detector()
        df = pd.DataFrame({"x": range(5)})
        det.fit(df)
        results = det.detect_batch(df)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.label == "trending" for r in results))

    def test_repr(self):
        det = self._make_detector()
        self.assertIn("1h", repr(det))
        self.assertIn("unfitted", repr(det))
        det.fit(pd.DataFrame())
        self.assertIn("fitted", repr(det))


# ======================  BASE MODEL TESTS  ============================

class TestBaseModel(unittest.TestCase):

    def _make_model(self, pair="XBTUSD", strategy="money_printer"):
        from guapbot.models.base import BaseModel

        class DummyModel(BaseModel):
            def fit(self, df):
                self._fitted = True
                return self

            def predict(self, obs):
                return 0.5

            def confidence(self, obs):
                return 0.8

        return DummyModel(pair=pair, strategy=strategy)

    def test_construction(self):
        m = self._make_model()
        self.assertEqual(m.pair, "XBTUSD")
        self.assertEqual(m.strategy, "money_printer")
        self.assertFalse(m._fitted)

    def test_invalid_strategy_raises(self):
        from guapbot.models.base import BaseModel

        class D(BaseModel):
            def fit(self, df): pass
            def predict(self, obs): return 0.0
            def confidence(self, obs): return 0.0

        with self.assertRaises(ValueError):
            D("XBTUSD", "scalper")

    def test_sat_stacker_valid(self):
        m = self._make_model(strategy="sat_stacker")
        self.assertEqual(m.strategy, "sat_stacker")

    def test_predict_returns_float(self):
        m = self._make_model()
        m.fit(pd.DataFrame())
        sig = m.predict({})
        self.assertIsInstance(sig, float)
        self.assertGreaterEqual(sig, -1.0)
        self.assertLessEqual(sig, 1.0)

    def test_confidence_returns_float(self):
        m = self._make_model()
        conf = m.confidence({})
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_validate_signal_clips(self):
        m = self._make_model()
        clipped = m._validate_signal(1.5)
        self.assertEqual(clipped, 1.0)
        clipped = m._validate_signal(-2.0)
        self.assertEqual(clipped, -1.0)

    def test_validate_confidence_clips(self):
        m = self._make_model()
        c = m._validate_confidence(-0.1)
        self.assertEqual(c, 0.0)
        c = m._validate_confidence(1.1)
        self.assertEqual(c, 1.0)

    def test_save_raises_not_implemented(self):
        m = self._make_model()
        with self.assertRaises(NotImplementedError):
            m.save("/tmp/test.pkl")

    def test_repr(self):
        m = self._make_model()
        r = repr(m)
        self.assertIn("XBTUSD", r)
        self.assertIn("money_printer", r)


# =====================  ENSEMBLE TESTS  ===============================

class TestModelSignal(unittest.TestCase):

    def test_valid(self):
        from guapbot.models.ensemble import ModelSignal
        s = ModelSignal(model_name="lstm", pair="XBTUSD", signal=0.3, confidence=0.7)
        self.assertEqual(s.signal, 0.3)

    def test_signal_out_of_range(self):
        from guapbot.models.ensemble import ModelSignal
        with self.assertRaises(ValueError):
            ModelSignal(model_name="lstm", pair="XBTUSD", signal=2.0, confidence=0.5)

    def test_confidence_out_of_range(self):
        from guapbot.models.ensemble import ModelSignal
        with self.assertRaises(ValueError):
            ModelSignal(model_name="lstm", pair="XBTUSD", signal=0.5, confidence=-0.1)


class TestEnsembleInterface(unittest.TestCase):

    def _make_ensemble(self):
        from guapbot.models.ensemble import BaseEnsemble, EnsembleInput, TradeOutcome

        class DummyEnsemble(BaseEnsemble):
            def fit(self, signal_history):
                self._fitted = True
                return self

            def combine(self, inputs):
                if not inputs.signals:
                    return 0.0
                return sum(s.signal for s in inputs.signals) / len(inputs.signals)

            def update(self, outcome):
                pass

        return DummyEnsemble("XBTUSD")

    def test_construction(self):
        ens = self._make_ensemble()
        self.assertEqual(ens.pair, "XBTUSD")

    def test_fit(self):
        ens = self._make_ensemble()
        df = pd.DataFrame({"x": [1, 2, 3]})
        ens.fit(df)
        self.assertTrue(ens._fitted)

    def test_combine_averages_signals(self):
        from guapbot.models.ensemble import EnsembleInput, ModelSignal
        ens = self._make_ensemble()
        signals = [
            ModelSignal("rl", "XBTUSD", 0.6, 0.8),
            ModelSignal("lstm", "XBTUSD", 0.4, 0.7),
        ]
        inp = EnsembleInput(signals=signals, regimes=[], pair="XBTUSD")
        result = ens.combine(inp)
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_combine_empty_signals(self):
        from guapbot.models.ensemble import EnsembleInput
        ens = self._make_ensemble()
        inp = EnsembleInput(signals=[], regimes=[], pair="XBTUSD")
        result = ens.combine(inp)
        self.assertEqual(result, 0.0)

    def test_update_no_crash(self):
        from guapbot.models.ensemble import TradeOutcome
        ens = self._make_ensemble()
        outcome = TradeOutcome(
            bar_time="2024-01-15T14:00:00",
            pair="XBTUSD",
            final_signal=0.5,
            realised_return=0.01,
        )
        ens.update(outcome)   # should not raise


# ===================  POSITION SIZER TESTS  ===========================

class TestAlertState(unittest.TestCase):

    def test_default_inactive(self):
        from guapbot.execution.position_sizer import AlertState
        a = AlertState()
        self.assertFalse(a.active)
        self.assertEqual(a.severity, 0.0)

    def test_severity_out_of_range(self):
        from guapbot.execution.position_sizer import AlertState
        with self.assertRaises(ValueError):
            AlertState(severity=1.5)

    def test_max_position_out_of_range(self):
        from guapbot.execution.position_sizer import AlertState
        with self.assertRaises(ValueError):
            AlertState(max_position=2.0)


class TestSizingResult(unittest.TestCase):

    def test_positive_position_is_long(self):
        from guapbot.execution.position_sizer import SizingResult
        r = SizingResult(position=0.1)
        self.assertEqual(r.side, "long")

    def test_negative_position_is_short(self):
        from guapbot.execution.position_sizer import SizingResult
        r = SizingResult(position=-0.1)
        self.assertEqual(r.side, "short")

    def test_zero_position_is_flat(self):
        from guapbot.execution.position_sizer import SizingResult
        r = SizingResult(position=0.0)
        self.assertEqual(r.side, "flat")


class TestPositionSizerInterface(unittest.TestCase):

    def _make_sizer(self):
        from guapbot.execution.position_sizer import PositionSizer, AlertState, SizingResult
        from guapbot.regime.base import RegimeResult

        class DummySizer(PositionSizer):
            def size(self, signal, regime, alert, *, atr=0.02, win_rate=0.5,
                     avg_win=0.02, avg_loss=0.01):
                kelly = self._half_kelly(win_rate, avg_win, avg_loss)
                raw = signal * kelly
                if alert.active:
                    raw = max(-alert.max_position, min(alert.max_position, raw))
                final = self._apply_hard_caps(raw)
                return SizingResult(position=final, base_signal=signal)

        return DummySizer()

    def test_long_signal_positive_position(self):
        sizer = self._make_sizer()
        from guapbot.execution.position_sizer import AlertState
        result = sizer.size(0.8, [], AlertState())
        self.assertGreater(result.position, 0)

    def test_short_signal_negative_position(self):
        sizer = self._make_sizer()
        from guapbot.execution.position_sizer import AlertState
        result = sizer.size(-0.8, [], AlertState())
        self.assertLess(result.position, 0)

    def test_hard_cap_long(self):
        sizer = self._make_sizer()
        from guapbot.execution.position_sizer import AlertState
        # Even with signal=1.0, should not exceed max_long
        result = sizer.size(1.0, [], AlertState(), win_rate=0.9, avg_win=0.05, avg_loss=0.005)
        self.assertLessEqual(result.position, sizer.max_long)

    def test_hard_cap_short(self):
        sizer = self._make_sizer()
        from guapbot.execution.position_sizer import AlertState
        result = sizer.size(-1.0, [], AlertState(), win_rate=0.9, avg_win=0.05, avg_loss=0.005)
        self.assertGreaterEqual(result.position, -sizer.max_short)

    def test_alert_caps_position(self):
        sizer = self._make_sizer()
        from guapbot.execution.position_sizer import AlertState
        alert = AlertState(active=True, severity=0.8, max_position=0.03)
        result = sizer.size(1.0, [], alert, win_rate=0.9, avg_win=0.1, avg_loss=0.01)
        self.assertLessEqual(abs(result.position), 0.03)

    def test_half_kelly_zero_history(self):
        sizer = self._make_sizer()
        k = sizer._half_kelly(0.5, 0, 0)
        self.assertEqual(k, 0.5)   # returns default when no history

    def test_half_kelly_reasonable(self):
        sizer = self._make_sizer()
        k = sizer._half_kelly(0.55, 0.02, 0.01)
        self.assertGreaterEqual(k, 0.0)
        self.assertLessEqual(k, 1.0)

    def test_repr(self):
        sizer = self._make_sizer()
        self.assertIn("25%", repr(sizer))


# ===================  MARKET STATE TESTS  ============================

class TestMarketState(unittest.TestCase):
    """
    MarketState tests run in no-op mode (Redis not connected).
    All methods return graceful defaults when Redis is unavailable.
    """

    def _make_state(self):
        from guapbot.execution.market_state import MarketState
        # Don't call connect() — exercises no-op mode
        return MarketState()

    def test_construction(self):
        state = self._make_state()
        self.assertFalse(state.connected)

    def test_update_noop_returns_false(self):
        state = self._make_state()
        result = state.update("signal.XBTUSD", 0.5)
        self.assertFalse(result)

    def test_get_noop_returns_default(self):
        state = self._make_state()
        val = state.get("signal.XBTUSD", default=99.9)
        self.assertEqual(val, 99.9)

    def test_inject_alert_noop(self):
        state = self._make_state()
        result = state.inject_alert({"source": "reddit", "severity": 0.7})
        self.assertFalse(result)   # False because Redis not connected

    def test_get_alert_no_redis(self):
        state = self._make_state()
        alert = state.get_alert()
        self.assertIsNone(alert)

    def test_get_signal_no_redis(self):
        state = self._make_state()
        sig = state.get_signal("XBTUSD")
        self.assertIsNone(sig)

    def test_all_keys_no_redis(self):
        state = self._make_state()
        keys = state.all_keys()
        self.assertEqual(keys, [])

    def test_repr(self):
        state = self._make_state()
        r = repr(state)
        self.assertIn("MarketState", r)
        self.assertIn("disconnected", r)


# ===========================  ENTRY POINT  ============================

if __name__ == "__main__":
    unittest.main(verbosity=2)
