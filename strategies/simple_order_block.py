# strategies/simple_order_block.py - FIXED VERSION
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
    WORKING Order Block strategy that actually generates signals!
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Core parameters with WORKING defaults
        self.symbol = kwargs.get('symbol', 'GER40.cash')
        self.ob_lookback = kwargs.get('ob_lookback', 5)
        self.sl_percent = kwargs.get('sl_percent', 0.01)  # 1% SL
        self.tp_percent = kwargs.get('tp_percent', 0.03)  # 3% TP

        # Enhanced filters - RELAXED for signal generation
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.rsi_min = kwargs.get('rsi_min', 25)  # More relaxed
        self.rsi_max = kwargs.get('rsi_max', 75)  # More relaxed
        self.volume_multiplier = kwargs.get('volume_multiplier', 1.1)  # Lower threshold
        self.volume_period = kwargs.get('volume_period', 20)

        # Bollinger settings
        self.bb_window = kwargs.get('bb_window', 20)
        self.bb_std = kwargs.get('bb_std', 1.5)

        logger.info("=== SimpleOrderBlockStrategy Initialized ===")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"SL: {self.sl_percent:.1%}, TP: {self.tp_percent:.1%}")
        logger.info(f"RSI Range: {self.rsi_min}-{self.rsi_max}")

    def generate_signals(self, df: pd.DataFrame, current_capital: float = None) -> \
    Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate trading signals - SIMPLIFIED to actually work!
        """
        logger.info(f"Generating signals for {len(df)} bars of data...")

        # 1. SIMPLE Order Block Detection (vectorized)
        body_size = abs(df['close'] - df['open'])
        range_size = df['high'] - df['low']

        # Strong bullish candles (simplified)
        is_bullish = df['close'] > df['open']
        big_body = body_size > body_size.rolling(self.ob_lookback).mean() * 1.2

        # 2. Trend Filter - Simple SMA
        sma20 = df['close'].rolling(20).mean()
        uptrend = df['close'] > sma20

        # 3. RSI Filter - More permissive
        rsi = vbt.RSI.run(df['close'], window=self.rsi_period).rsi
        rsi_ok = (rsi >= self.rsi_min) & (rsi <= self.rsi_max)

        # 4. Volume Filter - Use tick_volume if available
        if 'tick_volume' in df.columns:
            vol_ma = df['tick_volume'].rolling(self.volume_period).mean()
            volume_ok = df['tick_volume'] > (vol_ma * self.volume_multiplier)
        else:
            # Fallback: use range as volume proxy
            range_ma = range_size.rolling(self.volume_period).mean()
            volume_ok = range_size > (range_ma * self.volume_multiplier)

        # 5. COMBINE ALL FILTERS (relaxed combination)
        base_signals = is_bullish & big_body & uptrend
        enhanced_signals = base_signals & rsi_ok & volume_ok

        # Convert to integer series
        entries = enhanced_signals.fillna(False).astype(int)

        # 6. Stop Loss and Take Profit
        sl_stop = pd.Series(self.sl_percent, index=df.index)
        tp_stop = pd.Series(self.tp_percent, index=df.index)

        # 7. LOG RESULTS
        num_signals = entries.sum()
        logger.info(f"=== SIGNAL GENERATION RESULTS ===")
        logger.info(f"Total signals: {num_signals}")
        logger.info(f"Signal rate: {num_signals / len(df) * 100:.1f}% of bars")

        if num_signals > 0:
            signal_dates = df.index[entries > 0][:5]  # Show first 5
            for i, date in enumerate(signal_dates):
                price = df.loc[date, 'close']
                logger.info(f"Signal {i + 1}: {date}, Price: {price:.1f}")
        else:
            logger.warning("NO SIGNALS GENERATED! Strategy filters too restrictive.")

            # Debug info
            logger.info(f"Bullish candles: {is_bullish.sum()}")
            logger.info(f"Big body candles: {big_body.sum()}")
            logger.info(f"Uptrend bars: {uptrend.sum()}")
            logger.info(f"RSI OK bars: {rsi_ok.sum()}")
            logger.info(f"Volume OK bars: {volume_ok.sum()}")

        return entries, sl_stop, tp_stop