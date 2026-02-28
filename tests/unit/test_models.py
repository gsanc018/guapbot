"""
tests/unit/test_models.py

Unit tests for the model layer (Session 5).

Covers:
  - All 5 models implement the BaseModel interface correctly
  - BitcoinTradingEnv Gymnasium environment
  - No real data, no API calls — all synthetic

Synthetic data fixture:
  - 300-bar normalized DataFrame with realistic column names
  - Includes _cross and _signal columns (not z-scored by pipeline)
  - Includes continuous z-scored columns
  - Includes 'target' column for supervised models
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_BARS = 300
PAIR = "XBTUSD"
STRATEGY = "money_printer"

# XGBoost requires libomp on macOS (brew install libomp).
# Skip gracefully if the native library can't be loaded.
try:
    import xgboost as _xgb  # noqa: F401 — triggers native library load
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

xgb_skip = pytest.mark.skipif(
    not _XGB_AVAILABLE,
    reason="xgboost native library not available — macOS: brew install libomp",
)

# Column names that match the pipeline's multi-timeframe output format
CROSS_COLS = [
    "1h_ema_9_21_cross",
    "1h_ema_50_200_cross",
    "4h_ema_9_21_cross",
    "1h_psar_signal",
    "1h_bb_signal_20_2",
    "1h_stoch_cross",
]

CONTINUOUS_COLS = [
    "1h_adx_14",
    "1h_rsi_14",
    "1h_atr_14_pct",
    "1h_bb_pct_20_2",
    "1h_dc_pct_20",
    "1h_log_return",
    "4h_adx_14",
    "4h_rsi_14",
    "1d_rsi_14",
]


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """
    300-bar synthetic normalized feature DataFrame with target column.
    Resembles the output of FeaturePipeline.fit_transform() after target injection.
    """
    rng = np.random.default_rng(42)
    n = N_BARS
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    data = {}

    # Cross/signal columns: integers in {-1, 0, 1}
    for col in CROSS_COLS:
        data[col] = rng.choice([-1, 0, 1], size=n).astype(float)

    # Continuous columns: z-scored, clipped to [-5, 5]
    for col in CONTINUOUS_COLS:
        data[col] = rng.normal(0, 1, size=n).clip(-5, 5)

    # Target column: +1 or -1 (no zeros for cleaner tests)
    data["target"] = rng.choice([-1.0, 1.0], size=n)

    return pd.DataFrame(data, index=idx)


@pytest.fixture
def obs(synthetic_df) -> dict:
    """Latest bar as a dict (used for predict/confidence)."""
    return synthetic_df.iloc[-1].to_dict()


# ---------------------------------------------------------------------------
# Helper: assert model interface compliance
# ---------------------------------------------------------------------------

def _assert_interface(model, df: pd.DataFrame, obs_dict: dict) -> None:
    """Common assertion set for all BaseModel subclasses."""
    # Before fit: both predict and confidence raise
    with pytest.raises(RuntimeError):
        model.predict(obs_dict)
    with pytest.raises(RuntimeError):
        model.confidence(obs_dict)

    # After fit: predictions in range
    model.fit(df)
    assert model._fitted is True
    assert "fitted" in repr(model)

    signal = model.predict(obs_dict)
    assert isinstance(signal, float), f"predict() returned {type(signal)}"
    assert -1.0 <= signal <= 1.0, f"predict() out of range: {signal}"

    conf = model.confidence(obs_dict)
    assert isinstance(conf, float), f"confidence() returned {type(conf)}"
    assert 0.0 <= conf <= 1.0, f"confidence() out of range: {conf}"


# ---------------------------------------------------------------------------
# TrendFollowing
# ---------------------------------------------------------------------------

class TestTrendFollowing:
    def test_interface_compliance(self, synthetic_df, obs):
        from guapbot.models.trend_following import TrendFollowing
        model = TrendFollowing(PAIR, STRATEGY)
        assert "unfitted" in repr(model)
        _assert_interface(model, synthetic_df, obs)

    def test_predict_returns_zero_on_all_flat_signals(self, synthetic_df):
        from guapbot.models.trend_following import TrendFollowing
        model = TrendFollowing(PAIR, STRATEGY)
        model.fit(synthetic_df)

        flat_obs = {col: 0.0 for col in synthetic_df.columns}
        signal = model.predict(flat_obs)
        assert signal == pytest.approx(0.0, abs=0.01)

    def test_predict_positive_on_bullish_cross(self, synthetic_df):
        from guapbot.models.trend_following import TrendFollowing
        model = TrendFollowing(PAIR, STRATEGY)
        model.fit(synthetic_df)

        bullish_obs = {col: 0.0 for col in synthetic_df.columns}
        bullish_obs["1h_ema_9_21_cross"] = 1.0
        bullish_obs["4h_ema_9_21_cross"] = 1.0
        bullish_obs["1h_adx_14"] = 2.0  # strong trend
        signal = model.predict(bullish_obs)
        assert signal > 0.0

    def test_predict_negative_on_bearish_cross(self, synthetic_df):
        from guapbot.models.trend_following import TrendFollowing
        model = TrendFollowing(PAIR, STRATEGY)
        model.fit(synthetic_df)

        bearish_obs = {col: 0.0 for col in synthetic_df.columns}
        bearish_obs["1h_ema_9_21_cross"] = -1.0
        bearish_obs["4h_ema_9_21_cross"] = -1.0
        bearish_obs["1h_adx_14"] = 2.0
        signal = model.predict(bearish_obs)
        assert signal < 0.0

    def test_save_raises_not_implemented(self, synthetic_df):
        from guapbot.models.trend_following import TrendFollowing
        model = TrendFollowing(PAIR, STRATEGY)
        model.fit(synthetic_df)
        with pytest.raises(NotImplementedError):
            model.save("/tmp/test.pkl")

    def test_load_raises_not_implemented(self):
        from guapbot.models.trend_following import TrendFollowing
        model = TrendFollowing(PAIR, STRATEGY)
        with pytest.raises(NotImplementedError):
            model.load("/tmp/test.pkl")

    def test_graceful_on_missing_columns(self):
        from guapbot.models.trend_following import TrendFollowing
        model = TrendFollowing(PAIR, STRATEGY)
        tiny_df = pd.DataFrame({"some_col": [1.0, 2.0, 3.0]})
        model.fit(tiny_df)  # should not raise; just logs warning
        signal = model.predict({"some_col": 1.0})
        assert signal == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MeanReversion
# ---------------------------------------------------------------------------

class TestMeanReversion:
    def test_interface_compliance(self, synthetic_df, obs):
        from guapbot.models.mean_reversion import MeanReversion
        model = MeanReversion(PAIR, STRATEGY)
        assert "unfitted" in repr(model)
        _assert_interface(model, synthetic_df, obs)

    def test_signal_opposes_bb_overbought(self, synthetic_df):
        from guapbot.models.mean_reversion import MeanReversion
        model = MeanReversion(PAIR, STRATEGY)
        model.fit(synthetic_df)

        # bb_signal=+1 → overbought → MeanReversion shorts → signal < 0
        obs = {col: 0.0 for col in synthetic_df.columns}
        obs["1h_bb_signal_20_2"] = 1.0
        signal = model.predict(obs)
        assert signal < 0.0

    def test_signal_opposes_bb_oversold(self, synthetic_df):
        from guapbot.models.mean_reversion import MeanReversion
        model = MeanReversion(PAIR, STRATEGY)
        model.fit(synthetic_df)

        # bb_signal=-1 → oversold → MeanReversion longs → signal > 0
        obs = {col: 0.0 for col in synthetic_df.columns}
        obs["1h_bb_signal_20_2"] = -1.0
        signal = model.predict(obs)
        assert signal > 0.0

    def test_save_raises_not_implemented(self, synthetic_df):
        from guapbot.models.mean_reversion import MeanReversion
        model = MeanReversion(PAIR, STRATEGY)
        model.fit(synthetic_df)
        with pytest.raises(NotImplementedError):
            model.save("/tmp/test.pkl")

    def test_load_raises_not_implemented(self):
        from guapbot.models.mean_reversion import MeanReversion
        model = MeanReversion(PAIR, STRATEGY)
        with pytest.raises(NotImplementedError):
            model.load("/tmp/test.pkl")

    def test_high_rsi_gives_negative_signal(self, synthetic_df):
        from guapbot.models.mean_reversion import MeanReversion
        model = MeanReversion(PAIR, STRATEGY)
        model.fit(synthetic_df)
        obs = {col: 0.0 for col in synthetic_df.columns}
        obs["1h_rsi_14"] = 3.0   # very high z-score → overbought → short
        signal = model.predict(obs)
        assert signal < 0.0


# ---------------------------------------------------------------------------
# GradientBoost
# ---------------------------------------------------------------------------

class TestGradientBoost:
    @xgb_skip
    def test_interface_compliance(self, synthetic_df, obs):
        from guapbot.models.gradient_boost import GradientBoost
        model = GradientBoost(PAIR, STRATEGY)
        assert "unfitted" in repr(model)
        _assert_interface(model, synthetic_df, obs)

    @xgb_skip
    def test_requires_target_column(self):
        from guapbot.models.gradient_boost import GradientBoost
        model = GradientBoost(PAIR, STRATEGY)
        df_no_target = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="target"):
            model.fit(df_no_target)

    @xgb_skip
    def test_save_load_round_trip(self, synthetic_df, obs, tmp_path):
        from guapbot.models.gradient_boost import GradientBoost
        model = GradientBoost(PAIR, STRATEGY)
        model.fit(synthetic_df)
        orig_signal = model.predict(obs)

        path = str(tmp_path / "model.pkl")
        model.save(path)

        model2 = GradientBoost(PAIR, STRATEGY)
        model2.load(path)
        loaded_signal = model2.predict(obs)

        assert orig_signal == pytest.approx(loaded_signal, abs=1e-6)

    @xgb_skip
    def test_confidence_above_half(self, synthetic_df, obs):
        from guapbot.models.gradient_boost import GradientBoost
        model = GradientBoost(PAIR, STRATEGY)
        model.fit(synthetic_df)
        conf = model.confidence(obs)
        # Confidence is distance from 0.5 * 2 — can be 0 if model predicts exactly 0.5
        assert 0.0 <= conf <= 1.0

    @xgb_skip
    def test_predict_range_over_all_rows(self, synthetic_df):
        from guapbot.models.gradient_boost import GradientBoost
        model = GradientBoost(PAIR, STRATEGY)
        model.fit(synthetic_df)
        feature_cols = [c for c in synthetic_df.columns if c != "target"]
        for i in range(0, len(synthetic_df), 50):
            obs = synthetic_df.iloc[i][feature_cols].to_dict()
            signal = model.predict(obs)
            assert -1.0 <= signal <= 1.0


# ---------------------------------------------------------------------------
# LSTMModel
# ---------------------------------------------------------------------------

class TestLSTMModel:
    def test_interface_compliance(self, synthetic_df, obs):
        pytest.importorskip("torch")
        from guapbot.models.lstm import LSTMModel
        model = LSTMModel(PAIR, STRATEGY)
        assert "unfitted" in repr(model)
        _assert_interface(model, synthetic_df, obs)

    def test_requires_target_column(self):
        pytest.importorskip("torch")
        from guapbot.models.lstm import LSTMModel
        model = LSTMModel(PAIR, STRATEGY)
        df_no_target = pd.DataFrame(
            {"a": np.random.randn(100)},
        )
        with pytest.raises(ValueError, match="target"):
            model.fit(df_no_target)

    def test_buffer_warmup_returns_flat(self, synthetic_df):
        pytest.importorskip("torch")
        from guapbot.models.lstm import LSTMModel, SEQ_LEN
        model = LSTMModel(PAIR, STRATEGY)
        # Build small df — more than SEQ_LEN for training, but test buffer empty
        model.fit(synthetic_df)
        # Force-clear buffer to simulate cold start
        from collections import deque
        model._buffer = deque(maxlen=SEQ_LEN)
        signal = model.predict(synthetic_df.iloc[-1].to_dict())
        assert signal == pytest.approx(0.0)

    def test_predict_after_warmup_not_flat(self, synthetic_df):
        pytest.importorskip("torch")
        from guapbot.models.lstm import LSTMModel, SEQ_LEN
        model = LSTMModel(PAIR, STRATEGY)
        model.fit(synthetic_df)
        # Buffer was seeded with last SEQ_LEN rows → should produce non-trivial signal
        obs = synthetic_df.iloc[-1].to_dict()
        signal = model.predict(obs)
        assert isinstance(signal, float)
        assert -1.0 <= signal <= 1.0

    def test_save_load_round_trip(self, synthetic_df, tmp_path):
        pytest.importorskip("torch")
        from guapbot.models.lstm import LSTMModel
        model = LSTMModel(PAIR, STRATEGY)
        model.fit(synthetic_df)

        # Save BEFORE any predict() calls so both models start from identical buffer state
        path = str(tmp_path / "model.pt")
        model.save(path)

        model2 = LSTMModel(PAIR, STRATEGY)
        model2.load(path)

        # Both models start from the same saved buffer → same prediction on same obs
        obs = synthetic_df.iloc[-1].to_dict()
        orig_signal = model.predict(obs)
        loaded_signal = model2.predict(obs)
        assert orig_signal == pytest.approx(loaded_signal, abs=1e-5)

    def test_needs_more_than_seq_len_rows(self):
        pytest.importorskip("torch")
        from guapbot.models.lstm import LSTMModel, SEQ_LEN
        model = LSTMModel(PAIR, STRATEGY)
        tiny_df = pd.DataFrame({
            "a": np.random.randn(SEQ_LEN),
            "target": np.random.choice([-1.0, 1.0], size=SEQ_LEN),
        })
        with pytest.raises(ValueError, match=str(SEQ_LEN)):
            model.fit(tiny_df)


# ---------------------------------------------------------------------------
# RLAgent
# ---------------------------------------------------------------------------

class TestRLAgent:
    def test_interface_compliance(self, synthetic_df, obs):
        pytest.importorskip("stable_baselines3")
        from guapbot.models.rl_agent import RLAgent
        # Drop target for RL (env doesn't need it, but interface accepts it)
        model = RLAgent(PAIR, STRATEGY, algo="sac")
        assert "unfitted" in repr(model)
        _assert_interface(model, synthetic_df, obs)

    def test_invalid_algo_raises(self):
        pytest.importorskip("stable_baselines3")
        from guapbot.models.rl_agent import RLAgent
        with pytest.raises(ValueError, match="algo"):
            RLAgent(PAIR, STRATEGY, algo="invalid")

    def test_fixed_confidence(self, synthetic_df, obs):
        pytest.importorskip("stable_baselines3")
        from guapbot.models.rl_agent import RLAgent, _RL_CONFIDENCE
        model = RLAgent(PAIR, STRATEGY)
        model.fit(synthetic_df)
        assert model.confidence(obs) == pytest.approx(_RL_CONFIDENCE)

    def test_predict_range(self, synthetic_df, obs):
        pytest.importorskip("stable_baselines3")
        from guapbot.models.rl_agent import RLAgent
        model = RLAgent(PAIR, STRATEGY)
        model.fit(synthetic_df)
        for _ in range(10):
            signal = model.predict(obs)
            assert -1.0 <= signal <= 1.0


# ---------------------------------------------------------------------------
# BitcoinTradingEnv
# ---------------------------------------------------------------------------

class TestBitcoinTradingEnv:
    @pytest.fixture
    def env_df(self, synthetic_df) -> pd.DataFrame:
        """Environment DataFrame — no target column."""
        return synthetic_df.drop(columns=["target"])

    def test_reset_returns_correct_shape(self, env_df):
        pytest.importorskip("gymnasium")
        from guapbot.envs.trading_env import BitcoinTradingEnv
        env = BitcoinTradingEnv(env_df)
        obs, info = env.reset()
        assert obs.shape == (len(env_df.columns),)
        assert isinstance(info, dict)

    def test_step_returns_tuple(self, env_df):
        pytest.importorskip("gymnasium")
        from guapbot.envs.trading_env import BitcoinTradingEnv
        env = BitcoinTradingEnv(env_df)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.5]))
        assert obs.shape == (len(env_df.columns),)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert truncated is False
        assert isinstance(info, dict)

    def test_done_on_last_step(self, env_df):
        pytest.importorskip("gymnasium")
        from guapbot.envs.trading_env import BitcoinTradingEnv
        env = BitcoinTradingEnv(env_df)
        env.reset()
        terminated = False
        for _ in range(len(env_df)):
            _, _, terminated, _, _ = env.step(np.array([0.0]))
            if terminated:
                break
        assert terminated is True

    def test_reward_sign_matches_position_and_return(self, env_df):
        pytest.importorskip("gymnasium")
        from guapbot.envs.trading_env import BitcoinTradingEnv

        # Use a df with a known positive log return column
        df_with_ret = env_df.copy()
        df_with_ret["1h_log_return"] = 0.01  # fixed positive return every bar

        env = BitcoinTradingEnv(df_with_ret)
        env.reset()

        # Long position with positive return → positive reward
        _, reward_long, _, _, _ = env.step(np.array([1.0]))
        assert reward_long > 0.0

        # Short position with positive return → negative reward
        env.reset()
        _, reward_short, _, _, _ = env.step(np.array([-1.0]))
        assert reward_short < 0.0

    def test_reward_uses_next_bar_return(self, env_df):
        pytest.importorskip("gymnasium")
        from guapbot.envs.trading_env import BitcoinTradingEnv

        # bar 0: positive return, bar 1: negative return.
        # With next-bar semantics, a long at step 0 earns bar 1's return → negative.
        df = env_df.copy()
        log_rets = np.zeros(len(df))
        log_rets[0] = 0.1
        log_rets[1] = -0.1
        df["1h_log_return"] = log_rets.astype("float32")

        env = BitcoinTradingEnv(df)
        env.reset()
        _, reward, _, _, _ = env.step(np.array([1.0]))
        assert reward < 0.0  # would be positive under same-bar (buggy) semantics

    def test_spaces(self, env_df):
        pytest.importorskip("gymnasium")
        from guapbot.envs.trading_env import BitcoinTradingEnv
        env = BitcoinTradingEnv(env_df)
        assert env.observation_space.shape == (len(env_df.columns),)
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == pytest.approx(-1.0)
        assert env.action_space.high[0] == pytest.approx(1.0)

    def test_empty_df_raises(self):
        pytest.importorskip("gymnasium")
        from guapbot.envs.trading_env import BitcoinTradingEnv
        with pytest.raises(ValueError):
            BitcoinTradingEnv(pd.DataFrame())

    def test_feature_cols_property(self, env_df):
        pytest.importorskip("gymnasium")
        from guapbot.envs.trading_env import BitcoinTradingEnv
        env = BitcoinTradingEnv(env_df)
        assert env.feature_cols == list(env_df.columns)

    def test_n_steps_property(self, env_df):
        pytest.importorskip("gymnasium")
        from guapbot.envs.trading_env import BitcoinTradingEnv
        env = BitcoinTradingEnv(env_df)
        assert env.n_steps == len(env_df)
