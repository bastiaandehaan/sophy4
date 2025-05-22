# strategies/simple_order_block.py - Enhanced Version
from typing import Dict, Tuple, Optional, List, Any
import pandas as pd
import numpy as np
import vectorbt as vbt
from strategies import register_strategy
from strategies.base_strategy import BaseStrategy
from config import logger, get_risk_config, get_symbol_info
from utils.indicator_utils import calculate_bollinger_bands


@register_strategy
class SimpleOrderBlockStrategy(BaseStrategy):
    """
    Enhanced Order Block strategy with RSI and volume confirmation.

    Improvements over original:
    - RSI filtering to avoid overbought/oversold conditions
    - Volume confirmation for stronger signals
    - Dynamic position sizing based on recent performance
    - Better risk management integration
    """

    def __init__(self, symbol: str = "GER40.cash", ob_lookback: int = 5,
                 sl_percent: float = 0.01, tp_percent: float = 0.03,
                 rsi_period: int = 14, rsi_min: float = 35, rsi_max: float = 65,
                 volume_multiplier: float = 1.2, volume_period: int = 20,
                 dynamic_sizing: bool = True, bb_window: int = 20, bb_std: float = 1.5):
        super().__init__()

        # Core parameters
        self.symbol = symbol
        self.ob_lookback = ob_lookback
        self.sl_percent = sl_percent
        self.tp_percent = tp_percent

        # Enhanced filters - NEW
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min  # Avoid oversold conditions
        self.rsi_max = rsi_max  # Avoid overbought conditions
        self.volume_multiplier = volume_multiplier  # Volume threshold
        self.volume_period = volume_period

        # Position sizing - NEW
        self.dynamic_sizing = dynamic_sizing
        self.base_risk_pct = 0.01  # 1% base risk
        self.position_multiplier = 1.0

        # Bollinger confirmation - NEW
        self.bb_window = bb_window
        self.bb_std = bb_std

        # Performance tracking for dynamic sizing - NEW
        self.recent_trades = []
        self.max_recent_trades = 20

        # Get risk configuration
        self.risk_config = get_risk_config()
        self.symbol_info = get_symbol_info(symbol)

        # Log enhanced parameters
        logger.info("=== Enhanced SimpleOrderBlockStrategy Parameters ===")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Order Block Lookback: {ob_lookback}")
        logger.info(f"Stop Loss: {sl_percent:.2%}, Take Profit: {tp_percent:.2%}")
        logger.info(f"RSI Filter: {rsi_min}-{rsi_max} (period: {rsi_period})")
        logger.info(f"Volume Filter: {volume_multiplier}x over {volume_period} periods")
        logger.info(f"Dynamic Sizing: {dynamic_sizing}")
        logger.info(f"Bollinger Confirmation: {bb_window} period, {bb_std} std dev")
        logger.info("=" * 55)

    def _calculate_dynamic_risk(self) -> float:
        """Calculate dynamic risk based on recent performance."""
        if not self.dynamic_sizing or len(self.recent_trades) < 5:
            return self.base_risk_pct

        # Calculate recent performance metrics
        recent_pnl = [trade['pnl'] for trade in self.recent_trades[-10:]]
        win_rate = sum(1 for pnl in recent_pnl if pnl > 0) / len(recent_pnl)
        avg_pnl = np.mean(recent_pnl)

        # Adjust risk based on performance
        risk_multiplier = 1.0

        if win_rate > 0.6 and avg_pnl > 0:
            # Performing well - increase risk slightly
            risk_multiplier = 1.25
        elif win_rate < 0.4 or avg_pnl < 0:
            # Struggling - reduce risk
            risk_multiplier = 0.75

        # Apply bounds
        adjusted_risk = self.base_risk_pct * risk_multiplier
        max_risk = self.risk_config['max_risk_per_trade']

        return min(adjusted_risk, max_risk)

    def _update_trade_history(self, entry_price: float, exit_price: float,
                              position_size: float) -> None:
        """Update trade history for dynamic sizing."""
        pnl = (exit_price - entry_price) * position_size

        trade_record = {'entry_price': entry_price, 'exit_price': exit_price,
            'pnl': pnl, 'timestamp': pd.Timestamp.now()}

        self.recent_trades.append(trade_record)

        # Keep only recent trades
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades = self.recent_trades[-self.max_recent_trades:]

    def generate_signals(self, df: pd.DataFrame, current_capital: float = None) -> \
    Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate enhanced trading signals with multiple filters.
        """
        logger.info(
            "Generating enhanced order block signals with RSI and volume filters...")

        # 1. Basic order block detection (vectorized)
        body_size = abs(df['close'] - df['open'])
        range_size = df['high'] - df['low']
        avg_body = body_size.rolling(window=self.ob_lookback).mean()

        # Strong bullish candles
        is_bullish = df['close'] > df['open']
        strong_body = body_size > avg_body

        # 2. Trend filters
        sma20 = vbt.MA.run(df['close'], window=20).ma
        uptrend_sma = df['close'] > sma20

        # 3. Bollinger Bands confirmation - NEW
        upper_band, middle_band, lower_band = calculate_bollinger_bands(df['close'],
            window=self.bb_window, std_dev=self.bb_std)

        # Price above middle band indicates bullish momentum
        bb_bullish = df['close'] > middle_band

        # Price not too close to upper band (avoid overextension)
        bb_not_overextended = df['close'] < (upper_band * 0.95)

        # 4. RSI Filter - NEW 
        rsi = vbt.RSI.run(df['close'], window=self.rsi_period).rsi
        rsi_neutral = (rsi >= self.rsi_min) & (rsi <= self.rsi_max)

        logger.info(
            f"RSI filter: keeping {rsi_neutral.sum()} of {len(rsi_neutral)} bars")

        # 5. Volume Filter - NEW
        if 'tick_volume' in df.columns:
            volume_ma = df['tick_volume'].rolling(window=self.volume_period).mean()
            volume_confirmation = df['tick_volume'] > (
                        volume_ma * self.volume_multiplier)
        else:
            # Fallback: use range-based "volume" proxy
            range_ma = range_size.rolling(window=self.volume_period).mean()
            volume_confirmation = range_size > (range_ma * self.volume_multiplier)

        logger.info(
            f"Volume filter: keeping {volume_confirmation.sum()} of {len(volume_confirmation)} bars")

        # 6. Combine all filters
        base_signals = is_bullish & strong_body & uptrend_sma
        enhanced_signals = (
                    base_signals & bb_bullish & bb_not_overextended & rsi_neutral & volume_confirmation)

        entries = enhanced_signals.fillna(False).astype(int)

        # 7. Dynamic position sizing - NEW
        if self.dynamic_sizing:
            current_risk = self._calculate_dynamic_risk()
            self.position_multiplier = current_risk / self.base_risk_pct
            logger.info(
                f"Dynamic risk adjustment: {current_risk:.3f} (multiplier: {self.position_multiplier:.2f})")

        # 8. Stop-loss and take-profit
        sl_stop = pd.Series(self.sl_percent, index=df.index)
        tp_stop = pd.Series(self.tp_percent, index=df.index)

        # 9. Log results
        num_base_signals = base_signals.sum()
        num_enhanced_signals = entries.sum()

        logger.info(f"Signal filtering results:")
        logger.info(f"  Base signals (order block + trend): {num_base_signals}")
        logger.info(f"  After RSI filter: {(base_signals & rsi_neutral).sum()}")
        logger.info(
            f"  After volume filter: {(base_signals & rsi_neutral & volume_confirmation).sum()}")
        logger.info(f"  After Bollinger filter: {num_enhanced_signals}")
        logger.info(
            f"  Filter reduction: {(num_base_signals - num_enhanced_signals) / max(num_base_signals, 1):.1%}")

        if num_enhanced_signals > 0:
            signal_dates = df.index[entries > 0]
            for i, date in enumerate(signal_dates[:3]):  # Show first 3 signals
                price = df.loc[date, 'close']
                rsi_val = rsi.loc[date] if not pd.isna(rsi.loc[date]) else 0
                logger.info(
                    f"Signal {i + 1}: {date}, Price: {price:.2f}, RSI: {rsi_val:.1f}")
                logger.info(
                    f"  SL: {price * (1 - self.sl_percent):.2f}, TP: {price * (1 + self.tp_percent):.2f}")

            if num_enhanced_signals > 3:
                logger.info(f"... and {num_enhanced_signals - 3} more signals")

        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """Get enhanced default parameters for optimization."""
        from config import get_strategy_config

        # Get base config from unified system
        base_config = get_strategy_config("SimpleOrderBlockStrategy", timeframe)

        return {'symbol': [base_config.get('symbol', "GER40.cash")],
            'ob_lookback': [3, 5, 7, 10],
            'sl_percent': base_config.get('sl_percent_range', [0.01, 0.015, 0.02]),
            'tp_percent': base_config.get('tp_percent_range', [0.03, 0.04, 0.05]),

            # Enhanced parameters
            'rsi_period': [10, 14, 20], 'rsi_min': [30, 35, 40],
            'rsi_max': [60, 65, 70], 'volume_multiplier': [1.1, 1.2, 1.3, 1.5],
            'volume_period': [15, 20, 25], 'dynamic_sizing': [True, False],

            # Bollinger parameters
            'bb_window': [15, 20, 25], 'bb_std': [1.0, 1.5, 2.0],

            # Risk management
            'risk_per_trade': base_config.get('risk_per_trade_range',
                                              [0.005, 0.01, 0.015])}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """Get enhanced parameter descriptions."""
        return {'symbol': 'Trading symbol (e.g., GER40.cash, EURUSD)',
            'ob_lookback': 'Periods to look back for order block strength calculation',
            'sl_percent': 'Stop-loss as percentage of entry price',
            'tp_percent': 'Take-profit as percentage of entry price',

            # Enhanced descriptions
            'rsi_period': 'Period for RSI calculation',
            'rsi_min': 'Minimum RSI value to allow entry (avoid oversold)',
            'rsi_max': 'Maximum RSI value to allow entry (avoid overbought)',
            'volume_multiplier': 'Volume must be X times recent average',
            'volume_period': 'Period for volume average calculation',
            'dynamic_sizing': 'Adjust position size based on recent performance',

            # Bollinger descriptions
            'bb_window': 'Period for Bollinger Bands calculation',
            'bb_std': 'Standard deviations for Bollinger Bands',

            'risk_per_trade': 'Base risk per trade as percentage of capital'}

    @classmethod
    def get_performance_metrics(cls) -> List[str]:
        """Define performance metrics priority."""
        return ["sharpe_ratio", "calmar_ratio", "win_rate", "total_return",
                "max_drawdown", "profit_factor", "trades_count"]