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

    Version 2.1: Enhanced signal generation, LSTM input, and MT5 integration.
    """

    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "H1", window: int = 50,
                 lstm_threshold: float = 0.05, sl_fixed_percent: float = 0.01,
                 tp_fixed_percent: float = 0.025, use_trailing_stop: bool = False,
                 trailing_stop_percent: float = 0.015, risk_per_trade: float = 0.02,
                 confidence_level: float = 0.95, model_path: Optional[str] = None,
                 verbose_logging: bool = False, fib_lookback: int = 20,
                 initial_capital: float = INITIAL_CAPITAL):
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
            risk_per_trade (float): Risk per trade as portfolio percentage (0.02 = 2%).
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
        self.risk_manager = RiskManager(confidence_level=self.confidence_level,
                                        max_risk=self.risk_per_trade, max_daily_loss_percent=0.05,
                                        max_total_loss_percent=0.10, correlated_symbols=CORRELATED_SYMBOLS)

        print(f"\n==== OrderBlockLSTM Strategy v2.1 ({symbol}) ====")
        print("Enhanced MQL5-based order block detection with relaxed trading criteria")
        print(f"Parameters: window={window}, lstm_threshold={lstm_threshold}, "
              f"sl={sl_fixed_percent}, tp={tp_fixed_percent}, risk={risk_per_trade}")

        # Define mse function without decorator
        def mse(y_true, y_pred):
            return tf.keras.losses.mean_squared_error(y_true, y_pred)

        # Load LSTM model with improved error handling
        self.model = None
        self.scaler = MinMaxScaler()
        if model_path:
            model_abs_path = os.path.abspath(model_path)
            print(f"Attempting to load LSTM model from path: {model_abs_path}")
            try:
                if os.path.exists(model_abs_path):
                    self.model = load_model(model_abs_path, custom_objects={'mse': mse},
                                            compile=True)
                    print(f"✅ LSTM model successfully loaded from {model_abs_path}")
                    logger.info(f"LSTM model loaded from {model_abs_path}")
                else:
                    print(f"⚠️ Model file not found at {model_abs_path}")
                    logger.warning(f"Model file not found at {model_abs_path}")
            except Exception as e:
                print(f"⚠️ Failed to load LSTM model: {str(e)}")
                logger.error(f"Failed to load LSTM model: {str(e)}")
        else:
            logger.warning(f"No LSTM model provided for {symbol}, using fallback logic (LSTM pred = 0)")
            if self.verbose_logging:
                print(f"No LSTM model provided for {symbol}, using fallback logic")

    def get_symbol_info(self, symbol: str) -> Dict[str, float]:
        """
        Get symbol information with robust fallback values.

        Args:
            symbol (str): Trading symbol.

        Returns:
            Dict[str, float]: Symbol parameters with fallbacks.
        """
        import MetaTrader5 as mt5

        try:
            if not mt5.initialize():
                logger.warning(f"MT5 initialization failed for {symbol}.")
                return self._get_fallback_symbol_info(symbol)

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Symbol info not available for {symbol}.")
                mt5.shutdown()
                return self._get_fallback_symbol_info(symbol)

            tick_value = getattr(symbol_info, 'trade_tick_value', 0.0001 if symbol.startswith(('EUR', 'GBP', 'AUD')) else 10.0)
            point = symbol_info.point
            trade_tick_size = getattr(symbol_info, 'trade_tick_size', point)

            if symbol.startswith(('EUR', 'GBP', 'USD', 'AUD')):
                pip_value = 0.0001
                point = 0.00001
            else:
                pip_value = 1.0
                point = 0.1

            logger.info(f"Symbol info attributes for {symbol}: {dir(symbol_info)}")
            mt5.shutdown()
            return {
                "pip_value": pip_value,
                "spread": symbol_info.spread * point,
                "tick_value": tick_value,
                "contract_size": symbol_info.trade_contract_size,
                "point": point,
                "trade_tick_size": trade_tick_size
            }
        except Exception as e:
            logger.error(f"Error retrieving symbol info for {symbol}: {str(e)}")
            mt5.shutdown()
            return self._get_fallback_symbol_info(symbol)

    def _get_fallback_symbol_info(self, symbol: str) -> Dict[str, float]:
        """Fallback values based on symbol type."""
        if symbol.startswith(('EUR', 'GBP', 'USD', 'AUD')):
            return {"pip_value": 0.0001, "spread": 0.00002, "tick_value": 0.0001,
                    "contract_size": 100000, "point": 0.00001, "trade_tick_size": 0.00001}
        else:
            return {"pip_value": 1.0, "spread": 0.5, "tick_value": 1.0,
                    "contract_size": 1, "point": 0.1, "trade_tick_size": 0.1}

    def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect order blocks based on MQL5 article logic with relaxed criteria.

        Returns:
            List[OrderBlock]: List of detected order blocks.
        """
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {required_columns}")

        obs = []
        logger.info(f"Analyzing {len(df)} candles for order blocks ({self.symbol})")

        bullish_count = 0
        bearish_count = 0

        for i in range(len(df) - 4):
            if i + 3 >= len(df):
                continue

            c1, c2, c3, c4 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2], df.iloc[i + 3]

            if (c1['close'] > c1['open'] and
                    c2['close'] > c2['open'] and
                    c3['close'] < c3['open'] and
                    (c3['high'] - c3['low']) > (c2['high'] - c2['low']) * 1.1 and
                    c3['open'] < c2['close']):
                obs.append(OrderBlock(1, c3.name, c3['high'], c3['low']))
                bullish_count += 1
                if self.verbose_logging and bullish_count % 10 == 1:
                    logger.info(f"Bullish OB at {c3.name}: {c3['low']:.5f}-{c3['high']:.5f} ({self.symbol})")
                    print(f"Bullish OB at {c3.name}: {c3['low']:.5f}-{c3['high']:.5f} ({self.symbol})")

            elif (c1['close'] < c1['open'] and
                  c2['close'] < c2['open'] and
                  c3['close'] > c3['open'] and
                  (c3['high'] - c3['low']) > (c2['high'] - c2['low']) * 1.1 and
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
        return {'23.6%': swing_high - 0.236 * diff, '38.2%': swing_high - 0.382 * diff,
                '50.0%': swing_high - 0.5 * diff, '61.8%': swing_high - 0.618 * diff}

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def prepare_lstm_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare input for the LSTM model using enhanced features.

        Returns:
            np.ndarray: Normalized input array of shape (1, window, n_features).
        """
        if len(df) < self.window:
            logger.warning(f"Insufficient data for LSTM: need {self.window}, got {len(df)} ({self.symbol})")
            return np.zeros((1, self.window, 9))

        df = df.copy()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = self.calculate_rsi(df['close'], 14)

        feature_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'ma20', 'ma50', 'rsi']
        if 'spread' in df.columns:
            feature_columns.append('spread')

        recent_data = df.iloc[-self.window:][feature_columns].fillna(method='bfill').values
        normalized_data = self.scaler.fit_transform(recent_data)
        logger.info(f"LSTM input shape: {np.array([normalized_data]).shape}")
        return np.array([normalized_data.reshape(self.window, len(feature_columns))])

    def generate_signals(self, df: pd.DataFrame, current_capital: float = None) -> \
            Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate entry, stop-loss, and take-profit signals with relaxed MQL5-inspired criteria.

        Args:
            df (pd.DataFrame): DataFrame with OHLC data.
            current_capital (float, optional): Current account capital. Defaults to initial_capital.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (entries, sl_stop, tp_stop).
            entries: 1 for buy, -1 for sell, 0 for hold.
        """
        print(f"DEBUG: Analysing {len(df)} candles for OrderBlockLSTMStrategy with {self.symbol}")

        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {required_columns} for {self.symbol}")

        entries = pd.Series(0, index=df.index, dtype=int)
        sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
        tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

        if self.verbose_logging:
            print(f"\nGenerating signals for {len(df)} candles ({self.symbol})...")
        logger.info(f"Generating signals for {len(df)} candles ({self.symbol})")

        capital = current_capital or self.initial_capital

        max_value = capital
        if self.risk_manager.monitor_drawdown(capital, max_value):
            logger.warning(f"Drawdown limit exceeded for {self.symbol}, no new signals generated")
            return entries, sl_stop, tp_stop

        max_daily_loss = self.risk_manager.get_max_daily_loss(capital)
        max_total_loss = self.risk_manager.get_max_total_loss(capital)
        logger.info(f"FTMO limits for {self.symbol}: Max daily loss={max_daily_loss:.2f}, Max total loss={max_total_loss:.2f}")

        order_blocks = self.detect_order_blocks(df)
        print(f"DEBUG: Detected {len(order_blocks)} order blocks")

        if not order_blocks:
            if self.verbose_logging:
                print(f"No order blocks detected for {self.symbol}, returning empty signals")
            logger.info(f"No order blocks detected for {self.symbol}, returning empty signals")
            return entries, sl_stop, tp_stop

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

        try:
            time_threshold = df.index[-int(90 * 24)]
        except IndexError:
            time_threshold = df.index[0]

        current_price = df['close'].iloc[-1]
        if self.verbose_logging:
            print(f"Current price: {current_price:.5f}, time filter: OBs after {time_threshold} ({self.symbol})")
            print(f"LSTM prediction: {lstm_pred:.4f}")
        logger.info(f"Current price: {current_price:.5f}, time filter: OBs after {time_threshold} ({self.symbol})")

        returns = df['close'].pct_change().dropna()
        open_positions = {}

        signals_checked = 0
        recent_ob_count = 0
        signals_generated = 0

        for ob in order_blocks:
            signals_checked += 1

            if ob.time < time_threshold and not self.verbose_logging:
                continue

            recent_ob_count += 1

            past_data = df.loc[:ob.time].tail(self.fib_lookback)
            if past_data.empty:
                continue

            swing_high = past_data['high'].max()
            swing_low = past_data['low'].min()
            fib = self.calculate_fibonacci_levels(swing_high, swing_low)

            if self.verbose_logging and (signals_checked <= 10 or signals_checked % 50 == 0):
                print(f"\nOrder Block #{signals_checked}: {'Bullish' if ob.direction == 1 else 'Bearish'} at {ob.time} ({self.symbol})")
                print(f"  Price range: {ob.low:.5f}-{ob.high:.5f}")
                print(f"  Fibonacci levels: 23.6%={fib['23.6%']:.5f}, 38.2%={fib['38.2%']:.5f}, 50.0%={fib['50.0%']:.5f}, 61.8%={fib['61.8%']:.5f}")
                print(f"  Current price: {current_price:.5f}")

            trading_condition = False

            if ob.direction == 1:
                price_near_ob = abs(current_price - ob.low) / ob.low < 0.02
                fib_zone = (fib['61.8%'] * 0.9) <= current_price <= (fib['38.2%'] * 1.1)
                lstm_trend_ok = lstm_pred > -0.05
                trading_condition = price_near_ob and (fib_zone or lstm_trend_ok) and not ob.traded

                if self.verbose_logging and (signals_checked <= 5 or signals_checked % 50 == 0):
                    print(f"  Bullish trading conditions: price_near_ob={price_near_ob}, fib_zone={fib_zone}, lstm_trend_ok={lstm_trend_ok}")

            elif ob.direction == -1:
                price_near_ob = abs(current_price - ob.high) / ob.high < 0.02
                fib_zone = (fib['38.2%'] * 0.9) <= current_price <= (fib['61.8%'] * 1.1)
                lstm_trend_ok = lstm_pred < 0.05
                trading_condition = price_near_ob and (fib_zone or lstm_trend_ok) and not ob.traded

                if self.verbose_logging and (signals_checked <= 5 or signals_checked % 50 == 0):
                    print(f"  Bearish trading conditions: price_near_ob={price_near_ob}, fib_zone={fib_zone}, lstm_trend_ok={lstm_trend_ok}")

            if trading_condition:
                try:
                    size = self.risk_manager.calculate_adjusted_position_size(
                        capital=capital, returns=returns, symbol=self.symbol,
                        price=current_price, open_positions=open_positions)
                except Exception as e:
                    logger.error(f"Error calculating position size: {str(e)}")
                    size = 1.0

                if size > 0:
                    entries.iloc[-1] = ob.direction
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
        return {'symbol': SYMBOLS, 'window': [30, 50, 60],
                'lstm_threshold': [0.05, 0.1, 0.15],
                'sl_fixed_percent': [0.01, 0.015, 0.02],
                'tp_fixed_percent': [0.02, 0.025, 0.03], 'use_trailing_stop': [False, True],
                'trailing_stop_percent': [0.01, 0.015, 0.02],
                'risk_per_trade': [0.01, 0.02, 0.03],
                'confidence_level': [0.90, 0.95, 0.99], 'fib_lookback': [10, 20, 30],
                'initial_capital': [10000.0, 50000.0, 100000.0]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """
        Describe parameters for documentation.

        Returns:
            Dict[str, str]: Dictionary of parameter names and their descriptions.
        """
        return {'symbol': 'Trading symbol (e.g., XAUUSD)',
                'timeframe': 'Timeframe for trading (e.g., H1)',
                'window': 'Number of periods for LSTM input sequence',
                'lstm_threshold': 'Threshold for LSTM signals (0 to 1)',
                'sl_fixed_percent': 'Fixed stop-loss percentage',
                'tp_fixed_percent': 'Fixed take-profit percentage',
                'use_trailing_stop': 'Whether to use trailing stop (true/false)',
                'trailing_stop_percent': 'Trailing stop percentage',
                'risk_per_trade': 'Risk per trade as percentage of portfolio (0.02 = 2%)',
                'confidence_level': 'Confidence level for VaR calculation',
                'model_path': 'Path to pre-trained LSTM model (.h5 file)',
                'verbose_logging': 'Enable detailed logging for debugging',
                'fib_lookback': 'Lookback period for Fibonacci swing high/low calculation',
                'initial_capital': 'Initial account capital for risk management'}