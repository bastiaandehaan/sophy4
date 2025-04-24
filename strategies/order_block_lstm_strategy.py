from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow.keras.models import load_model
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow")
    exit(1)

from strategies.base_strategy import BaseStrategy
from strategies import register_strategy
from risk.risk_management import RiskManager
try:
    from config import logger, INITIAL_CAPITAL, SYMBOLS, CORRELATED_SYMBOLS, PIP_VALUES
except ImportError as e:
    print(f"Failed to import from config.py: {str(e)}")
    print("Please ensure config.py is in the project root and has no errors.")
    exit(1)


class OrderBlock:
    """Represents an order block zone in the market."""
    def __init__(self, direction: int, time: pd.Timestamp, high: float, low: float):
        self.direction = direction  # 1 = bullish, -1 = bearish
        self.time = time
        self.high = high
        self.low = low
        self.traded = False


@register_strategy
class OrderBlockLSTMStrategy(BaseStrategy):
    """
    Trading strategy based on order blocks, Fibonacci retracements, and LSTM predictions.

    Version 1.4: Multi-symbol support (EURUSD, GER40.cash, US30.cash, XAUUSD),
    FTMO-compliant RiskManager integration, and MT5 compatibility.
    """
    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "H1",
        window: int = 50,
        lstm_threshold: float = 0.3,
        sl_fixed_percent: float = 0.01,
        tp_fixed_percent: float = 0.02,
        use_trailing_stop: bool = False,
        trailing_stop_percent: float = 0.015,
        risk_per_trade: float = 0.01,
        confidence_level: float = 0.95,
        model_path: Optional[str] = None,
        verbose_logging: bool = False,
        fib_lookback: int = 20,
        initial_capital: float = INITIAL_CAPITAL
    ):
        """
        Initialize the OrderBlockLSTMStrategy.

        Args:
            symbol (str): Trading symbol (e.g., XAUUSD).
            timeframe (str): Timeframe (e.g., H1).
            window (int): Number of periods for LSTM input sequence.
            lstm_threshold (float): Threshold for LSTM signals (0 to 1).
            sl_fixed_percent (float): Fixed stop-loss percentage.
            tp_fixed_percent (float): Fixed take-profit percentage.
            use_trailing_stop (bool): Whether to use trailing stop.
            trailing_stop_percent (float): Trailing stop percentage.
            risk_per_trade (float): Risk per trade as portfolio percentage (0.01 = 1%).
            confidence_level (float): Confidence level for VaR calculation.
            model_path (Optional[str]): Path to pre-trained LSTM model (.h5 file).
            verbose_logging (bool): Enable detailed logging for debugging.
            fib_lookback (int): Lookback period for Fibonacci swing high/low.
            initial_capital (float): Initial account capital for risk management.
        """
        super().__init__()
        if symbol not in SYMBOLS:
            raise ValueError(f"Symbol {symbol} not in configured SYMBOLS: {SYMBOLS}")

        self.symbol = symbol
        self.timeframe = timeframe
        self.window = window
        self.lstm_threshold = lstm_threshold
        self.sl_fixed_percent = sl_fixed_percent
        self.tp_fixed_percent = tp_fixed_percent
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_percent = trailing_stop_percent
        self.risk_per_trade = risk_per_trade
        self.confidence_level = confidence_level
        self.verbose_logging = verbose_logging
        self.fib_lookback = fib_lookback
        self.initial_capital = initial_capital

        # Initialize RiskManager with FTMO-compliant settings
        self.risk_manager = RiskManager(
            confidence_level=self.confidence_level,
            max_risk=self.risk_per_trade,
            max_daily_loss_percent=0.05,  # FTMO: 5% daily loss
            max_total_loss_percent=0.10,  # FTMO: 10% total loss
            correlated_symbols=CORRELATED_SYMBOLS
        )

        print(f"\n==== OrderBlockLSTM Strategy v1.4 ({symbol}) ====")
        print("Multi-symbol support and FTMO-compliant RiskManager")
        print(f"Parameters: window={window}, lstm_threshold={lstm_threshold}, "
              f"sl={sl_fixed_percent}, tp={tp_fixed_percent}, risk={risk_per_trade}")

        # Define mse function without decorator
        def mse(y_true, y_pred):
            return tf.keras.losses.mean_squared_error(y_true, y_pred)

        # Load LSTM model with custom objects
        self.model = None
        self.scaler = MinMaxScaler()
        if model_path and os.path.exists(model_path):
            try:
                logger.debug(f"Attempting to load LSTM model from {model_path}")
                self.model = load_model(model_path, custom_objects={'mse': mse}, compile=True)
                logger.info(f"LSTM model loaded from {model_path}")
                if self.verbose_logging:
                    print(f"LSTM model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {str(e)}")
                print(f"Warning: Failed to load LSTM model: {str(e)}")
        else:
            logger.warning(f"No LSTM model provided for {symbol}, using fallback logic (LSTM pred = 0)")
            if self.verbose_logging:
                print(f"No LSTM model provided for {symbol}, using fallback logic")

    def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect order blocks based on MQL5 article logic.

        Returns:
            List[OrderBlock]: List of detected order blocks.
        """
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {required_columns}")

        obs = []
        logger.info(f"Analyzing {len(df)} candles for order blocks ({self.symbol})")
        if self.verbose_logging:
            print(f"Analyzing {len(df)} candles from {df.index[0]} to {df.index[-1]} ({self.symbol})")

        bullish_count = 0
        bearish_count = 0
        for i in range(len(df) - 3):
            if i + 2 >= len(df):
                continue
            c1, c2, c3 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2]

            # Bullish OB: Two bullish candles followed by a bearish candle
            if (c1['open'] < c1['close'] and
                c2['open'] < c2['close'] and
                c3['open'] > c3['close'] and
                c3['open'] < c2['close']):
                obs.append(OrderBlock(1, c3.name, c3['high'], c3['low']))
                bullish_count += 1
                if self.verbose_logging and bullish_count % 10 == 1:
                    logger.info(f"Bullish OB at {c3.name}: {c3['low']:.5f}-{c3['high']:.5f} ({self.symbol})")
                    print(f"Bullish OB at {c3.name}: {c3['low']:.5f}-{c3['high']:.5f} ({self.symbol})")

            # Bearish OB: Two bearish candles followed by a bullish candle
            elif (c1['open'] > c1['close'] and
                  c2['open'] > c2['close'] and
                  c3['open'] < c3['close'] and
                  c3['open'] > c2['close']):
                obs.append(OrderBlock(-1, c3.name, c3['high'], c3['low']))
                bearish_count += 1
                if self.verbose_logging and bearish_count % 10 == 1:
                    logger.info(f"Bearish OB at {c3.name}: {c3['low']:.5f}-{c3['high']:.5f} ({self.symbol})")
                    print(f"Bearish OB at {c3.name}: {c3['low']:.5f}-{c3['high']:.5f} ({self.symbol})")

        logger.info(f"Order Blocks detected: {bullish_count} bullish, {bearish_count} bearish ({self.symbol})")
        if self.verbose_logging:
            print(f"Order Blocks detected: {bullish_count} bullish, {bearish_count} bearish, total {len(obs)} ({self.symbol})")

        if not obs and self.verbose_logging:
            logger.warning(f"No order blocks detected for {self.symbol}. Check data or pattern logic.")
            print(f"No order blocks detected for {self.symbol}. Verify data or pattern logic.")

        return obs

    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            swing_high (float): Swing high price.
            swing_low (float): Swing low price.

        Returns:
            Dict[str, float]: Dictionary with Fibonacci levels (23.6%, 38.2%, 50.0%, 61.8%).
        """
        diff = swing_high - swing_low
        return {
            '23.6%': swing_high - 0.236 * diff,
            '38.2%': swing_high - 0.382 * diff,
            '50.0%': swing_high - 0.5 * diff,
            '61.8%': swing_high - 0.618 * diff
        }

    def prepare_lstm_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare input for the LSTM model using OHLC data.

        Returns:
            np.ndarray: Normalized input array of shape (1, window, 4).
        """
        if len(df) < self.window:
            logger.warning(f"Insufficient data for LSTM: need {self.window}, got {len(df)} ({self.symbol})")
            return np.zeros((1, self.window, 4))

        recent_data = df.iloc[-self.window:][['open', 'high', 'low', 'close']].values
        normalized_data = self.scaler.fit_transform(recent_data.reshape(-1, 4)).reshape(self.window, 4)
        return np.array([normalized_data])

    def generate_signals(self, df: pd.DataFrame, current_capital: float = None) -> Tuple[
        pd.Series, pd.Series, pd.Series
    ]:
        """
        Generate entries, stop-loss, and take-profit signals.

        Args:
            df (pd.DataFrame): DataFrame with OHLC data.
            current_capital (float, optional): Current account capital. Defaults to initial_capital.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]:
            (entries, sl_stop, tp_stop).
            entries: 1 for buy, -1 for sell, 0 for hold.
        """
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {required_columns} for {self.symbol}")

        entries = pd.Series(0, index=df.index, dtype=int)
        sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
        tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

        if self.verbose_logging:
            print(f"\nGenerating signals for {len(df)} candles ({self.symbol})...")
        logger.info(f"Generating signals for {len(df)} candles ({self.symbol})")

        # Use provided capital or fallback to initial_capital
        capital = current_capital or self.initial_capital

        # FTMO compliance: Check drawdown and loss limits
        max_value = capital  # Placeholder; ideally track max portfolio value via MT5
        if self.risk_manager.monitor_drawdown(capital, max_value):
            logger.warning(f"Drawdown limit exceeded for {self.symbol}, no new signals generated")
            return entries, sl_stop, tp_stop

        max_daily_loss = self.risk_manager.get_max_daily_loss(capital)
        max_total_loss = self.risk_manager.get_max_total_loss(capital)
        logger.info(f"FTMO limits for {self.symbol}: Max daily loss={max_daily_loss:.2f}, Max total loss={max_total_loss:.2f}")

        # Detect order blocks
        order_blocks = self.detect_order_blocks(df)
        if not order_blocks:
            if self.verbose_logging:
                print(f"No order blocks detected for {self.symbol}, returning empty signals")
            logger.info(f"No order blocks detected for {self.symbol}, returning empty signals")
            return entries, sl_stop, tp_stop

        # Prepare LSTM prediction
        lstm_pred = 0.0
        if self.model is not None:
            try:
                X_in = self.prepare_lstm_input(df)
                lstm_pred = self.model.predict(X_in, verbose=0)[0][0]
                if self.verbose_logging:
                    print(f"LSTM prediction: {lstm_pred:.4f}, threshold: {self.lstm_threshold} ({self.symbol})")
                logger.info(f"LSTM prediction: {lstm_pred:.4f}, threshold: {self.lstm_threshold} ({self.symbol})")
            except Exception as e:
                logger.error(f"LSTM prediction failed for {self.symbol}: {str(e)}")
                if self.verbose_logging:
                    print(f"LSTM prediction failed for {self.symbol}: {str(e)}")
        else:
            if self.verbose_logging:
                print(f"No LSTM model available for {self.symbol}, using default LSTM value of 0")
            logger.warning(f"No LSTM model available for {self.symbol}, using default LSTM value of 0")

        # Time filter: only consider recent order blocks (last 14 days for H1)
        try:
            time_threshold = df.index[-int(14 * 24)]  # 14 days * 24 H1 candles
        except IndexError:
            time_threshold = df.index[0]  # Fallback to start if data is too short

        current_price = df['close'].iloc[-1]
        if self.verbose_logging:
            print(f"Current price: {current_price:.5f}, time filter: OBs after {time_threshold} ({self.symbol})")
        logger.info(f"Current price: {current_price:.5f}, time filter: OBs after {time_threshold} ({self.symbol})")

        # Calculate returns for VaR
        returns = df['close'].pct_change().dropna()

        # Placeholder for open positions (to be integrated with MT5)
        open_positions = {}  # e.g., {'XAUUSD': 2, 'EURUSD': 1}

        signals_checked = 0
        recent_ob_count = 0
        signals_generated = 0
        for ob in order_blocks:
            signals_checked += 1
            if ob.time < time_threshold:
                continue
            recent_ob_count += 1

            # Calculate Fibonacci levels with lookback
            past_data = df.loc[:ob.time].tail(self.fib_lookback)
            if past_data.empty:
                continue
            swing_high = past_data['high'].max()
            swing_low = past_data['low'].min()
            fib = self.calculate_fibonacci_levels(swing_high, swing_low)

            if self.verbose_logging and (signals_checked % 10 == 1 or recent_ob_count <= 5):
                print(f"\nOrder Block #{signals_checked}: {'Bullish' if ob.direction == 1 else 'Bearish'} at {ob.time} ({self.symbol})")
                print(f"  Price range: {ob.low:.5f}-{ob.high:.5f}")
                print(f"  Fibonacci levels: 23.6%={fib['23.6%']:.5f}, 38.2%={fib['38.2%']:.5f}, 50.0%={fib['50.0%']:.5f}, 61.8%={fib['61.8%']:.5f}")
                logger.info(f"Order Block #{signals_checked}: {'Bullish' if ob.direction == 1 else 'Bearish'} at {ob.time} ({self.symbol})")
                logger.info(f"  Price range: {ob.low:.5f}-{ob.high:.5f}")
                logger.info(f"  Fibonacci levels: 23.6%={fib['23.6%']:.5f}, 38.2%={fib['38.2%']:.5f}, 50.0%={fib['50.0%']:.5f}, 61.8%={fib['61.8%']:.5f}")

            # Check Fibonacci zones and LSTM for signals
            in_fib_zone = False
            lstm_ok = False
            if ob.direction == 1:  # Bullish
                in_618_zone = fib['61.8%'] <= current_price <= ob.high
                in_50_zone = fib['50.0%'] <= current_price <= ob.high
                in_382_zone = fib['38.2%'] <= current_price <= ob.high
                in_236_zone = fib['23.6%'] <= current_price <= ob.high
                in_fib_zone = in_618_zone or in_50_zone or in_382_zone or in_236_zone
                lstm_ok = lstm_pred > self.lstm_threshold
            elif ob.direction == -1:  # Bearish
                in_618_zone = ob.low <= current_price <= fib['61.8%']
                in_50_zone = ob.low <= current_price <= fib['50.0%']
                in_382_zone = ob.low <= current_price <= fib['38.2%']
                in_236_zone = ob.low <= current_price <= fib['23.6%']
                in_fib_zone = in_618_zone or in_50_zone or in_382_zone or in_236_zone
                lstm_ok = lstm_pred < -self.lstm_threshold

            if in_fib_zone and lstm_ok and not ob.traded:
                # Calculate position size with RiskManager
                size = self.risk_manager.calculate_adjusted_position_size(
                    capital=capital,
                    returns=returns,
                    symbol=self.symbol,
                    price=current_price,
                    open_positions=open_positions
                )
                if size > 0:
                    entries.iloc[-1] = ob.direction  # 1 for buy, -1 for sell
                    signals_generated += 1
                    ob.traded = True
                    if self.verbose_logging:
                        print(f"✅ {'Bullish' if ob.direction == 1 else 'Bearish'} signal generated at {df.index[-1]}, size={size:.2f} ({self.symbol})")
                    logger.info(f"❗ {'Bullish' if ob.direction == 1 else 'Bearish'} signal generated at {df.index[-1]}, size={size:.2f} ({self.symbol})")

        if self.verbose_logging:
            print(f"\nSignal generation results for {self.symbol}:")
            print(f"  Order Blocks analyzed: {signals_checked}")
            print(f"  Recent Order Blocks: {recent_ob_count}")
            print(f"  Signals generated: {signals_generated}")
        logger.info(f"Signal generation results for {self.symbol}: {signals_generated} signals from {recent_ob_count} recent OBs")

        # Clear VaR cache to prevent memory buildup
        self.risk_manager.clear_cache()

        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """
        Return default parameters for optimization.

        Args:
            timeframe (str): Timeframe (e.g., H1).

        Returns:
            Dict[str, List[Any]]: Dictionary of parameter names and their possible values.
        """
        return {
            'symbol': SYMBOLS,
            'window': [30, 50, 60],
            'lstm_threshold': [0.2, 0.3, 0.4],
            'sl_fixed_percent': [0.01, 0.015, 0.02],
            'tp_fixed_percent': [0.02, 0.03, 0.04],
            'use_trailing_stop': [False, True],
            'trailing_stop_percent': [0.01, 0.015, 0.02],
            'risk_per_trade': [0.005, 0.01, 0.015],
            'confidence_level': [0.90, 0.95, 0.99],
            'fib_lookback': [10, 20, 30],
            'initial_capital': [10000.0, 50000.0, 100000.0]
        }

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """
        Describe parameters for documentation.

        Returns:
            Dict[str, str]: Dictionary of parameter names and their descriptions.
        """
        return {
            'symbol': 'Trading symbol (e.g., XAUUSD)',
            'timeframe': 'Timeframe for trading (e.g., H1)',
            'window': 'Number of periods for LSTM input sequence',
            'lstm_threshold': 'Threshold for LSTM signals (0 to 1)',
            'sl_fixed_percent': 'Fixed stop-loss percentage',
            'tp_fixed_percent': 'Fixed take-profit percentage',
            'use_trailing_stop': 'Whether to use trailing stop (true/false)',
            'trailing_stop_percent': 'Trailing stop percentage',
            'risk_per_trade': 'Risk per trade as percentage of portfolio (0.01 = 1%)',
            'confidence_level': 'Confidence level for VaR calculation',
            'model_path': 'Path to pre-trained LSTM model (.h5 file)',
            'verbose_logging': 'Enable detailed logging for debugging',
            'fib_lookback': 'Lookback period for Fibonacci swing high/low calculation',
            'initial_capital': 'Initial account capital for risk management'
        }