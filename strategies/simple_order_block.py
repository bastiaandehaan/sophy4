from typing import Dict, Tuple
import pandas as pd
import vectorbt as vbt
from strategies import register_strategy
from strategies.base_strategy import BaseStrategy
from config import logger
from utils.indicator_utils import calculate_bollinger_bands

@register_strategy
class SimpleOrderBlockStrategy(BaseStrategy):
    """
    Simplified Order Block strategy with Bollinger Bands for trend confirmation.
    """

    def __init__(self, symbol: str = "GER40.cash", ob_lookback: int = 5,
                 sl_percent: float = 0.01, tp_percent: float = 0.03):
        super().__init__()
        self.symbol = symbol
        self.ob_lookback = ob_lookback  # Lookback period for order blocks
        self.sl_percent = sl_percent    # Stop-loss percentage
        self.tp_percent = tp_percent    # Take-profit percentage

        # Fixed risk settings for robustness
        self.fixed_risk_pct = 0.01      # 1% risk per trade
        self.position_multiplier = 1.0  # No multiplication, conservative

        # Log parameters
        logger.info("=== SimpleOrderBlockStrategy Parameters ===")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Order Block Lookback: {ob_lookback}")
        logger.info(f"Stop Loss: {sl_percent:.2%}")
        logger.info(f"Take Profit: {tp_percent:.2%}")
        logger.info(f"Fixed Risk %: {self.fixed_risk_pct:.2%}")
        logger.info(f"Position Multiplier: {self.position_multiplier}x")
        logger.info("==========================================")

    def generate_signals(self, df: pd.DataFrame,
                         current_capital: float = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate trading signals based on order block detection with trend confirmation.
        """
        logger.info("Generating signals with order block and Bollinger Bands criteria...")

        # 1. Calculate body size and range (vectorized)
        body_size = abs(df['close'] - df['open'])
        range_size = df['high'] - df['low']

        # 2. Calculate rolling average body size
        avg_body = body_size.rolling(window=self.ob_lookback).mean()

        # 3. Trend filter using SMA20
        sma20 = vbt.MA.run(df['close'], window=20).ma
        uptrend_sma = df['close'] > sma20

        # 4. Trend filter using Bollinger Bands (price above middle band for uptrend)
        upper_band, middle_band, lower_band = calculate_bollinger_bands(df['close'], window=20, std_dev=1.5)
        uptrend_bb = df['close'] > middle_band

        # 5. Simplified order block: Bullish candle with strong body + trend confirmation
        is_bullish = df['close'] > df['open']
        strong_body = body_size > avg_body  # Basic strength filter
        entries = is_bullish & strong_body & uptrend_sma & uptrend_bb
        entries = entries.fillna(False).astype(int)

        # 6. Stop-loss and take-profit (vectorized)
        sl_stop = pd.Series(self.sl_percent, index=df.index)
        tp_stop = pd.Series(self.tp_percent, index=df.index)

        # 7. Log results
        num_signals = entries.sum()
        logger.info(f"Total signals: {num_signals}")

        if num_signals > 0:
            signal_dates = df.index[entries > 0]
            for i, date in enumerate(signal_dates[:5]):  # First 5 signals
                price = df.loc[date, 'close']
                logger.info(f"Signal {i + 1}: {date}, Price: {price:.2f}, SL: {price * (1 - self.sl_percent):.2f}, TP: {price * (1 + self.tp_percent):.2f}")

            if num_signals > 5:
                logger.info(f"... and {num_signals - 5} more signals")

        return entries, sl_stop, tp_stop