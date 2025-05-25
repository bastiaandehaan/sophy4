# strategies/simple_order_block.py - FIXED VERSION
# Replace the __init__ method in your SimpleOrderBlockStrategy class

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
    FIXED Order Block Strategy - Now properly accepts filter parameters
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Core parameters (working correctly)
        self.symbol = kwargs.get('symbol', 'GER40.cash')
        self.ob_lookback = kwargs.get('ob_lookback', 5)
        self.sl_percent = kwargs.get('sl_percent', 0.01)
        self.tp_percent = kwargs.get('tp_percent', 0.03)

        # RSI filters (already working)
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.rsi_min = kwargs.get('rsi_min', 25)
        self.rsi_max = kwargs.get('rsi_max', 75)
        self.volume_multiplier = kwargs.get('volume_multiplier', 1.1)
        self.volume_period = kwargs.get('volume_period', 20)

        # CRITICAL MISSING PARAMETERS - Add these to enable frequency control
        self.use_rejection_wicks = kwargs.get('use_rejection_wicks', True)
        self.use_session_filter = kwargs.get('use_session_filter', False)
        self.use_htf_confirmation = kwargs.get('use_htf_confirmation',
                                               True)  # KEY BOTTLENECK
        self.min_wick_ratio = kwargs.get('min_wick_ratio', 0.3)  # KEY BOTTLENECK
        self.stress_threshold = kwargs.get('stress_threshold', 2.2)  # KEY BOTTLENECK

        # Additional filter controls
        self.use_volume_filter = kwargs.get('use_volume_filter', True)
        self.min_body_ratio = kwargs.get('min_body_ratio', 1.5)
        self.trend_strength_min = kwargs.get('trend_strength_min', 1.2)

        # Bollinger settings
        self.bb_window = kwargs.get('bb_window', 20)
        self.bb_std = kwargs.get('bb_std', 1.5)

        # Session hours (if session filter enabled)
        self.session_start = kwargs.get('session_start', 8)
        self.session_end = kwargs.get('session_end', 18)

        # Risk parameters
        self.risk_per_trade = kwargs.get('risk_per_trade', 0.01)

        # Log the ACTUAL filter configuration
        logger.info("=== UPDATED SimpleOrderBlockStrategy ===")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"🎯 FREQUENCY CONTROL FILTERS:")
        logger.info(
            f"   HTF Confirmation: {self.use_htf_confirmation} ({'ENABLED' if self.use_htf_confirmation else 'DISABLED'})")
        logger.info(
            f"   Stress Threshold: {self.stress_threshold} ({'TIGHT' if self.stress_threshold < 2.5 else 'RELAXED'})")
        logger.info(
            f"   Min Wick Ratio: {self.min_wick_ratio} ({'STRICT' if self.min_wick_ratio > 0.25 else 'PERMISSIVE'})")
        logger.info(
            f"   Session Filter: {self.use_session_filter} ({'ENABLED' if self.use_session_filter else 'DISABLED'})")
        logger.info(
            f"   Rejection Wicks: {self.use_rejection_wicks} ({'REQUIRED' if self.use_rejection_wicks else 'OPTIONAL'})")
        logger.info(f"📊 RSI Range: {self.rsi_min}-{self.rsi_max}")
        logger.info(f"💰 Risk/Trade: {self.risk_per_trade:.1%}")

    def detect_rejection_wicks(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect rejection wicks - now properly controlled by min_wick_ratio parameter
        """
        if not self.use_rejection_wicks:
            # If rejection wicks disabled, return all True (no filter)
            return pd.Series(True, index=df.index)

        body_size = abs(df['close'] - df['open'])
        upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
        lower_wick = df[['close', 'open']].min(axis=1) - df['low']

        # Use the configurable min_wick_ratio parameter
        rejection_signal = ((lower_wick > body_size * self.min_wick_ratio) & (
                    upper_wick < body_size * 0.4) & (df['close'] > df['open']))

        return rejection_signal

    def is_active_session(self, df: pd.DataFrame) -> pd.Series:
        """
        Session filter - now properly controlled by use_session_filter parameter
        """
        if not self.use_session_filter:
            # If session filter disabled, return all True (trade 24/7)
            return pd.Series(True, index=df.index)

        if not hasattr(df.index, 'hour'):
            return pd.Series(True, index=df.index)

        # Use configurable session hours
        active_hours = (df.index.hour >= self.session_start) & (
                    df.index.hour <= self.session_end)
        return active_hours

    def get_htf_confirmation(self, df: pd.DataFrame) -> bool:
        """
        HTF confirmation - now properly controlled by use_htf_confirmation parameter
        """
        if not self.use_htf_confirmation:
            # If HTF confirmation disabled, always return True (allow all signals)
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
        Market stress detection - now properly controlled by stress_threshold parameter
        """
        # Calculate ATR-based volatility
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean()
        atr_ma = atr.rolling(50).mean()
        vol_ratio = atr / atr_ma

        # Use the configurable stress_threshold parameter
        low_stress = vol_ratio < self.stress_threshold

        return low_stress

    def generate_signals(self, df: pd.DataFrame, current_capital: float = None) -> \
    Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate signals with proper filter parameter usage
        """
        logger.info(f"Generating signals with UPDATED filter parameters...")
        logger.info(f"   Data bars: {len(df)}")

        # BASELINE signals (core order block logic)
        body_size = abs(df['close'] - df['open'])
        is_bullish = df['close'] > df['open']
        big_body = body_size > body_size.rolling(
            self.ob_lookback).mean() * self.min_body_ratio

        sma20 = df['close'].rolling(20).mean()
        uptrend = df['close'] > sma20

        rsi = vbt.RSI.run(df['close'], window=self.rsi_period).rsi
        rsi_ok = (rsi >= self.rsi_min) & (rsi <= self.rsi_max)

        # Volume filter (with configurable parameter)
        if self.use_volume_filter:
            if 'tick_volume' in df.columns:
                vol_ma = df['tick_volume'].rolling(self.volume_period).mean()
                volume_ok = df['tick_volume'] > (vol_ma * self.volume_multiplier)
            else:
                range_size = df['high'] - df['low']
                range_ma = range_size.rolling(self.volume_period).mean()
                volume_ok = range_size > (range_ma * self.volume_multiplier)
        else:
            volume_ok = pd.Series(True, index=df.index)

        # BASELINE signals (before filters)
        base_signals = is_bullish & big_body & uptrend & rsi_ok & volume_ok

        # APPLY CONFIGURABLE FILTERS
        enhanced_filters = pd.Series(True, index=df.index)

        # Filter 1: Rejection Wicks (configurable)
        rejection_wicks = self.detect_rejection_wicks(df)
        enhanced_filters = enhanced_filters & rejection_wicks
        logger.info(
            f"   After rejection wick filter: {(base_signals & enhanced_filters).sum()} signals")

        # Filter 2: Session Filter (configurable)
        active_session = self.is_active_session(df)
        enhanced_filters = enhanced_filters & active_session
        logger.info(
            f"   After session filter: {(base_signals & enhanced_filters).sum()} signals")

        # Filter 3: HTF Confirmation (configurable - KEY BOTTLENECK)
        htf_bullish = self.get_htf_confirmation(df)
        if not htf_bullish:
            logger.info("   HTF bearish - blocking all signals")
            enhanced_filters = enhanced_filters & False
        else:
            logger.info("   HTF bullish - signals allowed")

        # Filter 4: Market Stress Filter (configurable - KEY BOTTLENECK)
        low_stress = self.detect_market_stress(df)
        enhanced_filters = enhanced_filters & low_stress
        logger.info(
            f"   After stress filter: {(base_signals & enhanced_filters).sum()} signals")

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

        logger.info(f"🎯 SIGNAL GENERATION RESULTS:")
        logger.info(f"   Baseline signals: {num_baseline}")
        logger.info(f"   After ALL filters: {num_enhanced}")
        logger.info(f"   Filter reduction: {filter_reduction:.1f}%")
        logger.info(f"   Expected trades/year: {num_enhanced * (365 / len(df)):.0f}")

        return entries, sl_stop, tp_stop

    @classmethod
    def get_optimization_params(cls) -> Dict[str, List[Any]]:
        """
        UPDATED optimization parameters including frequency control filters
        """
        return {# Core parameters
            'ob_lookback': [3, 5, 7], 'sl_percent': [0.008, 0.01, 0.012, 0.015],
            'tp_percent': [0.025, 0.03, 0.035, 0.04],

            # RSI parameters
            'rsi_min': [15, 20, 25, 30], 'rsi_max': [70, 75, 80, 85],

            # FREQUENCY CONTROL PARAMETERS (the missing piece!)
            'use_htf_confirmation': [False, True],  # KEY: False = 2x more signals
            'stress_threshold': [2.0, 2.5, 2.8, 3.0, 3.5],  # KEY: Higher = more signals
            'min_wick_ratio': [0.1, 0.15, 0.2, 0.25, 0.3],  # KEY: Lower = more signals
            'use_session_filter': [False, True],  # KEY: False = 24/7 trading
            'use_rejection_wicks': [False, True],  # KEY: False = no wick requirement

            # Volume parameters
            'volume_multiplier': [0.8, 0.9, 1.0, 1.1, 1.2],
            'use_volume_filter': [False, True],

            # Body size parameters
            'min_body_ratio': [1.0, 1.2, 1.5, 1.8, 2.0]}