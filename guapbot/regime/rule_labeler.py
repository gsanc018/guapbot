"""
guapbot.regime.rule_labeler
---------------------------
Rule-based regime labeller for HMM training data.

Applies deterministic thresholds to technical features to produce
a 'label' column.  The HMM then trains on these labels, learning
the multivariate feature distributions per regime and modelling
regime transitions — which the rules alone cannot do.

Intraday (1h / 4h) labels:  trending | ranging | volatile
Daily labels:                bullish  | bearish  | neutral

Thresholds are intentionally conservative — when in doubt, the bar
is labelled 'ranging' or 'neutral' (the safe/default state).

Usage:
    from guapbot.regime.rule_labeler import RuleLabeler

    labeler = RuleLabeler(timeframe="1h")
    labeled_df = labeler.label_dataframe(features_df)
    # labeled_df now has a 'label' column — pass to HMMDetector.fit()
"""

from __future__ import annotations

import pandas as pd

from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------

# Intraday: volatile wins if either condition is met
_VOLATILE_ATR_PCT = 0.025        # ATR > 2.5% of close
_VOLATILE_BB_WIDTH = 0.08        # Bollinger width > 8% of mid

# Intraday: trending if ADX is strong (and not volatile)
_TRENDING_ADX = 25.0

# Daily: RSI thresholds for macro bias
_BULLISH_RSI = 55.0
_BEARISH_RSI = 45.0

# Daily: price vs EMA-21 to confirm direction
_PRICE_VS_EMA_COL = "price_vs_ema21"   # from features/technical.py


class RuleLabeler:
    """
    Deterministic regime labeller using technical indicator thresholds.

    No API calls, no training, no randomness.  Results are reproducible
    and instant.  The HMM trained on these labels learns the statistical
    structure of each regime and can generalise beyond the rules.
    """

    def __init__(self, timeframe: str) -> None:
        """
        Args:
            timeframe: '1h', '4h', or 'daily'
        """
        if timeframe not in ("1h", "4h", "daily"):
            raise ValueError(f"timeframe must be '1h', '4h', or 'daily', got {timeframe}")
        self.timeframe = timeframe

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regime rules to every row of *df*.

        Args:
            df: Feature DataFrame from FeaturePipeline.
                Must include at minimum the columns checked in _label_row.

        Returns:
            df.copy() with 'label' column added (str, no NaN).
        """
        if self.timeframe == "daily":
            labels = self._label_daily(df)
        else:
            labels = self._label_intraday(df)

        result = df.copy()
        result["label"] = labels

        counts = result["label"].value_counts().to_dict()
        log.info(
            "Rule labelling complete | timeframe=%s bars=%d distribution=%s",
            self.timeframe, len(df), counts,
        )
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _label_intraday(self, df: pd.DataFrame) -> pd.Series:
        """
        1h / 4h regime rules.

        Priority order (first match wins):
            1. volatile  — high ATR% or wide Bollinger bands
            2. trending  — ADX above threshold
            3. ranging   — default
        """
        volatile = pd.Series(False, index=df.index)
        trending = pd.Series(False, index=df.index)

        if "atr_14_pct" in df.columns:
            volatile |= df["atr_14_pct"] > _VOLATILE_ATR_PCT
        if "bb_width_20_2" in df.columns:
            volatile |= df["bb_width_20_2"] > _VOLATILE_BB_WIDTH
        if "adx_14" in df.columns:
            trending = df["adx_14"] > _TRENDING_ADX

        labels = pd.Series("ranging", index=df.index)
        labels[trending & ~volatile] = "trending"
        labels[volatile] = "volatile"
        return labels

    def _label_daily(self, df: pd.DataFrame) -> pd.Series:
        """
        Daily macro regime rules.

        Priority order:
            1. bullish  — RSI > 55 and price above EMA-21 (if available)
            2. bearish  — RSI < 45 and price below EMA-21 (if available)
            3. neutral  — default
        """
        labels = pd.Series("neutral", index=df.index)

        if "rsi_14" not in df.columns:
            log.warning("rsi_14 not found — all daily bars labelled 'neutral'")
            return labels

        rsi = df["rsi_14"]
        bullish_rsi = rsi > _BULLISH_RSI
        bearish_rsi = rsi < _BEARISH_RSI

        # Confirm with price vs EMA if available
        if _PRICE_VS_EMA_COL in df.columns:
            ema_signal = df[_PRICE_VS_EMA_COL]
            bullish = bullish_rsi & (ema_signal > 0)
            bearish = bearish_rsi & (ema_signal < 0)
        else:
            bullish = bullish_rsi
            bearish = bearish_rsi

        labels[bullish] = "bullish"
        labels[bearish] = "bearish"
        return labels
