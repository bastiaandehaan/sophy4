# strategies/simple_order_block.py - COMPLETE BALANCED VERSION
from typing import Dict, Tuple, Optional, List, Any
import pandas as pd
import numpy as np
import vectorbt as vbt
from strategies import register_strategy
from strategies.base_strategy import BaseStrategy
from config import logger, config_manager
from utils.indicator_utils import calculate_bollinger_bands


@register_strategy
class SimpleOrderBlockStrategy(BaseStrategy):
    """
    BALANCED Order Block Strategy - Optimized for 42-44% Win Rate + More Trades
    Target: 60-80 trades with 42-44% win rate vs 26 trades with 46% win rate
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Core parameters
        self.symbol = kwargs.get('symbol', 'GER40.cash')
        self.ob_lookback = kwargs.get('ob_lookback', 5)
        self.sl_percent = kwargs.get('sl_percent', 0.01)
        self.tp_percent = kwargs.get('tp_percent', 0.03)

        # Enhanced filters - RELAXED for more trades
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.rsi_min = kwargs.get('rsi_min', 25)
        self.rsi_max = kwargs.get('rsi_max', 75)
        self.volume_multiplier = kwargs.get('volume_multiplier', 1.1)
        self.volume_period = kwargs.get('volume_period', 20)

        # BALANCED PERFORMANCE BOOSTERS
        self.use_rejection_wicks = kwargs.get('use_rejection_wicks', True)
        self.use_session_filter = kwargs.get('use_session_filter',
                                             False)  # DISABLED for more trades
        self.use_htf_confirmation = kwargs.get('use_htf_confirmation', True)
        self.min_wick_ratio = kwargs.get('min_wick_ratio', 0.3)  # RELAXED from 0.4
        self.stress_threshold = kwargs.get('stress_threshold', 2.2)  # RELAXED from 1.8

        # Bollinger settings
        self.bb_window = kwargs.get('bb_window', 20)
        self.bb_std = kwargs.get('bb_std', 1.5)

        logger.info("=== BALANCED SimpleOrderBlockStrategy ===")
        logger.info(f"TARGET: 42-44% Win Rate with 60-80 Trades")
        logger.info(
            f"Rejection Wicks: {self.use_rejection_wicks} (ratio: {self.min_wick_ratio})")
        logger.info(f"Session Filter: {self.use_session_filter}")
        logger.info(f"HTF Confirmation: {self.use_htf_confirmation}")
        logger.info(f"Stress Threshold: {self.stress_threshold}")

    def detect_rejection_wicks(self, df: pd.DataFrame) -> pd.Series:
        """
        RELAXED Rejection Wick Detection - More permissive for more signals
        """
        body_size = abs(df['close'] - df['open'])
        upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
        lower_wick = df[['close', 'open']].min(axis=1) - df['low']

        # RELAXED: Lower minimum wick ratio (0.3 vs 0.4)
        rejection_signal = ((
                                        lower_wick > body_size * self.min_wick_ratio) &  # Relaxed wick requirement
                            (
                                        upper_wick < body_size * 0.4) &  # Slightly relaxed upper wick
                            (df['close'] > df['open'])  # Bullish close
        )

        return rejection_signal

    def is_active_session(self, df: pd.DataFrame) -> pd.Series:
        """
        EXPANDED Session Filter - More trading hours (if enabled)
        """
        if not hasattr(df.index, 'hour'):
            return pd.Series(True, index=df.index)

        # EXPANDED: 08:00-18:00 UTC (vs original 13:00-17:00)
        # Includes London morning + overlap + NY afternoon
        active_hours = (df.index.hour >= 8) & (df.index.hour <= 18)

        return active_hours

    def get_htf_confirmation(self, df: pd.DataFrame) -> bool:
        """
        Higher Timeframe Confirmation - KEPT (works well)
        """
        if not self.use_htf_confirmation:
            return True

        try:
            # Get H4 data for confirmation
            htf_data = self.fetch_historical_data(self.symbol, 'H4', days=30)

            if htf_data is None or len(htf_data) < 20:
                return True  # Default to allow if can't get HTF data

            # Simple HTF trend: Close above 20 SMA
            htf_sma = htf_data['close'].rolling(20).mean()
            htf_bullish = htf_data['close'].iloc[-1] > htf_sma.iloc[-1]

            return htf_bullish

        except Exception as e:
            logger.debug(f"HTF confirmation failed: {e}")
            return True  # Default to allow

    def detect_market_stress(self, df: pd.DataFrame) -> pd.Series:
        """
        RELAXED Market Stress Detection - Allow more volatility
        """
        # Calculate ATR-based volatility
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean()
        atr_ma = atr.rolling(50).mean()
        vol_ratio = atr / atr_ma

        # RELAXED: Allow higher volatility (2.2 vs 1.8)
        low_stress = vol_ratio < self.stress_threshold

        return low_stress

    def generate_signals(self, df: pd.DataFrame, current_capital: float = None) -> \
    Tuple[pd.Series, pd.Series, pd.Series]:
        """
        BALANCED signal generation - Quality vs Quantity optimized
        """
        logger.info(f"Generating BALANCED signals for {len(df)} bars...")

        # ORIGINAL FILTERS (proven baseline)
        body_size = abs(df['close'] - df['open'])
        is_bullish = df['close'] > df['open']
        big_body = body_size > body_size.rolling(
            self.ob_lookback).mean() * 1.15  # SLIGHTLY relaxed

        sma20 = df['close'].rolling(20).mean()
        uptrend = df['close'] > sma20

        rsi = vbt.RSI.run(df['close'], window=self.rsi_period).rsi
        rsi_ok = (rsi >= self.rsi_min) & (rsi <= self.rsi_max)

        if 'tick_volume' in df.columns:
            vol_ma = df['tick_volume'].rolling(self.volume_period).mean()
            volume_ok = df['tick_volume'] > (vol_ma * self.volume_multiplier)
        else:
            range_size = df['high'] - df['low']
            range_ma = range_size.rolling(self.volume_period).mean()
            volume_ok = range_size > (range_ma * self.volume_multiplier)

        # BASELINE SIGNALS
        base_signals = is_bullish & big_body & uptrend & rsi_ok & volume_ok

        # SELECTIVE PERFORMANCE BOOSTERS
        enhanced_filters = pd.Series(True, index=df.index)

        # Booster #1: Rejection Wicks (KEPT - high impact)
        if self.use_rejection_wicks:
            rejection_wicks = self.detect_rejection_wicks(df)
            enhanced_filters = enhanced_filters & rejection_wicks
            logger.info(f"Rejection wick filter: {rejection_wicks.sum()} bars pass")

        # Booster #2: Session Filter (DISABLED for more trades)
        if self.use_session_filter:
            active_session = self.is_active_session(df)
            enhanced_filters = enhanced_filters & active_session
            logger.info(f"Session filter: {active_session.sum()} bars in active hours")
        else:
            logger.info("Session filter DISABLED - trading all hours")

        # Booster #3: HTF Confirmation (KEPT - high quality filter)
        htf_bullish = self.get_htf_confirmation(df)
        if not htf_bullish:
            logger.info("HTF bearish - reducing signals")
            enhanced_filters = enhanced_filters & False  # Block all signals
        else:
            logger.info("HTF bullish - signals allowed")

        # Booster #4: Market Stress Filter (RELAXED)
        low_stress = self.detect_market_stress(df)
        enhanced_filters = enhanced_filters & low_stress
        logger.info(
            f"Low stress filter: {low_stress.sum()} bars pass (threshold: {self.stress_threshold})")

        # COMBINE ALL FILTERS
        final_signals = base_signals & enhanced_filters
        entries = final_signals.fillna(False).astype(int)

        # Stop Loss and Take Profit
        sl_stop = pd.Series(self.sl_percent, index=df.index)
        tp_stop = pd.Series(self.tp_percent, index=df.index)

        # RESULTS LOGGING
        num_baseline = base_signals.sum()
        num_enhanced = entries.sum()
        filter_reduction = (num_baseline - num_enhanced) / max(num_baseline, 1) * 100

        logger.info(f"BALANCED SIGNAL FILTERING RESULTS:")
        logger.info(f"   Baseline signals: {num_baseline}")
        logger.info(f"   Enhanced signals: {num_enhanced}")
        logger.info(f"   Filter reduction: {filter_reduction:.1f}%")
        logger.info(f"   Target: 42-44% win rate with 60-80 trades")

        if num_enhanced > 0:
            signal_dates = df.index[entries > 0][:3]
            for i, date in enumerate(signal_dates):
                price = df.loc[date, 'close']
                logger.info(f"   Signal {i + 1}: {date}, Price: {price:.1f}")

        return entries, sl_stop, tp_stop

    @classmethod
    def get_optimization_params(cls) -> Dict[str, List[Any]]:
        """
        Parameters for optimization - BALANCED approach
        """
        return {'ob_lookback': [3, 5, 7], 'sl_percent': [0.008, 0.01, 0.012, 0.015],
            'tp_percent': [0.025, 0.03, 0.035, 0.04], 'rsi_min': [20, 25, 30],
            'rsi_max': [70, 75, 80], 'volume_multiplier': [1.0, 1.1, 1.2],
            'min_wick_ratio': [0.25, 0.3, 0.35], 'stress_threshold': [2.0, 2.2, 2.5],
            'use_session_filter': [False, True], 'use_rejection_wicks': [True],
            # Always keep this
            'use_htf_confirmation': [True]  # Always keep this
        }