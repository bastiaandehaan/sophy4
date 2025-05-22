# strategies/order_block_lstm_strategy.py - Robust Version
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import os
import json
import logging

# TensorFlow warnings suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning,ignore::FutureWarning'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
    import keras_tuner as kt

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM strategy will use fallback mode.")

import sys
import os
import vectorbt as vbt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import logger, get_strategy_config, get_symbol_info
from strategies.base_strategy import BaseStrategy
from strategies import register_strategy
from utils.error_handling import safe_execute, ErrorResult, TradingError


# Enhanced Error Classes
class ModelLoadError(TradingError):
    """Model loading specific errors."""
    pass


class ModelPredictionError(TradingError):
    """Model prediction specific errors."""
    pass


class FallbackStrategyError(TradingError):
    """Fallback strategy errors."""
    pass


# Attention Layer for LSTM
if TENSORFLOW_AVAILABLE:
    class AttentionLayer(Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight',
                                     shape=(input_shape[-1], 1),
                                     initializer='random_normal', trainable=True)
            self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                     initializer='zeros', trainable=True)
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
            alpha = tf.keras.backend.softmax(e, axis=1)
            context = inputs * alpha
            context = tf.keras.backend.sum(context, axis=1)
            return context


@register_strategy
class OrderBlockLSTMStrategy(BaseStrategy):
    """
    Enhanced Order Block strategy with LSTM predictions and robust error handling.

    Features:
    - Proper model loading with validation
    - Comprehensive fallback strategy
    - Detailed error reporting
    - Performance monitoring
    - Safe configuration handling
    """

    def __init__(self, symbol: str = "GER40.cash", timeframe: str = "H1",
                 ob_lookback: int = 20, ob_strength: float = 2.0,
                 lstm_threshold: float = 0.4, sl_fixed_percent: float = 0.02,
                 tp_fixed_percent: float = 0.045, use_trailing_stop: bool = True,
                 trailing_stop_percent: float = 0.01, risk_per_trade: float = 0.0075,
                 model_confidence_threshold: float = 0.6):
        super().__init__()

        # Core parameters
        self.symbol = symbol
        self.timeframe = timeframe
        self.ob_lookback = ob_lookback
        self.ob_strength = ob_strength
        self.lstm_threshold = lstm_threshold
        self.sl_fixed_percent = sl_fixed_percent
        self.tp_fixed_percent = tp_fixed_percent
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_percent = trailing_stop_percent
        self.risk_per_trade = risk_per_trade
        self.model_confidence_threshold = model_confidence_threshold

        # Model state
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.model_validated = False
        self.seq_len = 50  # Must match training
        self.feature_columns = ['open', 'high', 'low', 'close', 'tick_volume']

        # Error tracking
        self.model_errors = []
        self.fallback_mode = False
        self.last_model_check = None

        # Performance tracking
        self.prediction_accuracy = []
        self.model_confidence_history = []

        logger.info("=== OrderBlockLSTMStrategy Initialization ===")
        logger.info(f"Symbol: {symbol}, Timeframe: {timeframe}")
        logger.info(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")

        # Initialize model with robust error handling
        self._initialize_model()

        logger.info(
            f"Strategy Mode: {'LSTM + OrderBlock' if self.model_loaded else 'Fallback OrderBlock'}")
        logger.info("=" * 50)

    @safe_execute(fallback_value=False, log_errors=True)
    def _initialize_model(self) -> bool:
        """Initialize LSTM model with comprehensive error handling."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - using fallback strategy")
            self.fallback_mode = True
            return False

        try:
            # Construct model path
            model_filename = f"lstm_{self.symbol}_{self.timeframe}.h5"
            possible_paths = [Path("./trainedh5") / model_filename,
                              Path("./models") / model_filename,
                              Path(".") / model_filename,
                              Path("./strategies/models") / model_filename]

            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break

            if not model_path:
                raise ModelLoadError(
                    f"LSTM model not found for {self.symbol}_{self.timeframe}. "
                    f"Searched paths: {[str(p) for p in possible_paths]}")

            logger.info(f"Loading LSTM model from: {model_path}")

            # Load model with custom objects
            custom_objects = {}
            if TENSORFLOW_AVAILABLE:
                custom_objects['AttentionLayer'] = AttentionLayer

            self.model = load_model(str(model_path), custom_objects=custom_objects)

            # Validate model
            validation_result = self._validate_model()
            if not validation_result:
                raise ModelLoadError("Model validation failed")

            self.model_loaded = True
            self.model_validated = True
            self.last_model_check = datetime.now()

            logger.info("✅ LSTM model loaded and validated successfully")
            return True

        except Exception as e:
            error_msg = f"Model initialization failed: {str(e)}"
            logger.error(error_msg)
            self.model_errors.append({'timestamp': datetime.now(), 'error': error_msg,
                'type': type(e).__name__})

            # Enable fallback mode
            self.fallback_mode = True
            logger.warning("🔄 Switching to fallback strategy mode")
            return False

    def _validate_model(self) -> bool:
        """Validate loaded model with test data."""
        if not self.model:
            return False

        try:
            # Create dummy data for validation
            dummy_input = np.random.random((1, self.seq_len, len(self.feature_columns)))
            prediction = self.model.predict(dummy_input, verbose=0)

            # Check prediction shape and values
            if prediction.shape != (1, 1):
                logger.error(
                    f"Invalid prediction shape: {prediction.shape}, expected: (1, 1)")
                return False

            if not np.isfinite(prediction[0, 0]):
                logger.error(f"Model returned invalid prediction: {prediction[0, 0]}")
                return False

            logger.info(
                f"Model validation successful - test prediction: {prediction[0, 0]:.4f}")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False

    @safe_execute(fallback_value=pd.Series(0, index=[]))
    def detect_order_block(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect order blocks with enhanced validation.
        """
        if len(df) < self.ob_lookback + 1:
            logger.warning(
                f"Insufficient data for order block detection: {len(df)} < {self.ob_lookback + 1}")
            return pd.Series(0, index=df.index)

        signals = pd.Series(0, index=df.index)

        try:
            for i in range(self.ob_lookback, len(df)):
                window = df.iloc[i - self.ob_lookback:i]
                current_candle = df.iloc[i]

                # Validate data quality
                if window.isnull().any().any() or pd.isna(current_candle['close']):
                    continue

                # Calculate metrics with error handling
                try:
                    if 'tick_volume' in window.columns:
                        avg_volume = window['tick_volume'].mean()
                        current_volume = current_candle.get('tick_volume', avg_volume)
                    else:
                        avg_volume = (window['high'] - window['low']).mean()
                        current_volume = current_candle['high'] - current_candle['low']

                    avg_range = (window['high'] - window['low']).mean()
                    current_range = current_candle['high'] - current_candle['low']

                    # Bullish order block detection
                    is_bullish = current_candle['close'] > current_candle['open']
                    strong_body = (current_candle['close'] - current_candle[
                        'open']) > avg_range * self.ob_strength
                    high_volume = current_volume > avg_volume * self.ob_strength

                    if is_bullish and strong_body and high_volume:
                        signals.iloc[i] = 1

                except Exception as e:
                    logger.debug(f"Error in order block calculation at index {i}: {e}")
                    continue

            logger.info(
                f"Order block detection completed: {signals.sum()} signals found")
            return signals

        except Exception as e:
            logger.error(f"Order block detection failed: {str(e)}")
            return pd.Series(0, index=df.index)

    @safe_execute(fallback_value=None)
    def _get_lstm_prediction(self, df: pd.DataFrame) -> Optional[float]:
        """Get LSTM prediction with comprehensive error handling."""
        if not self.model_loaded or not self.model:
            return None

        try:
            # Prepare recent data
            if len(df) < self.seq_len:
                logger.warning(
                    f"Insufficient data for LSTM: {len(df)} < {self.seq_len}")
                return None

            # Get recent data
            recent_data = df.tail(self.seq_len).copy()

            # Feature engineering (matching training)
            recent_data['ma20'] = recent_data['close'].rolling(
                window=min(20, len(recent_data))).mean()
            recent_data['ma50'] = recent_data['close'].rolling(
                window=min(50, len(recent_data))).mean()
            recent_data['price_movement'] = recent_data['close'] - recent_data['open']
            recent_data['candle_range'] = recent_data['high'] - recent_data['low']

            # Fill NaN values
            recent_data = recent_data.fillna(method='bfill').fillna(method='ffill')

            # Prepare features
            feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'ma20',
                            'ma50', 'price_movement', 'candle_range']

            # Use available columns only
            available_cols = [col for col in feature_cols if col in recent_data.columns]
            if len(available_cols) < 4:
                logger.error(f"Insufficient features available: {available_cols}")
                return None

            # Normalize data (simple min-max scaling)
            feature_data = recent_data[available_cols].values

            # Simple normalization
            data_min = feature_data.min(axis=0)
            data_max = feature_data.max(axis=0)
            data_range = data_max - data_min
            data_range[data_range == 0] = 1  # Avoid division by zero

            normalized_data = (feature_data - data_min) / data_range

            # Reshape for LSTM
            X = normalized_data.reshape(1, self.seq_len, len(available_cols))

            # Get prediction
            prediction = self.model.predict(X, verbose=0)[0, 0]

            # Validate prediction
            if not np.isfinite(prediction):
                logger.warning(f"Invalid LSTM prediction: {prediction}")
                return None

            # Calculate confidence (based on prediction magnitude)
            confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 scale
            self.model_confidence_history.append(confidence)

            logger.debug(
                f"LSTM prediction: {prediction:.4f}, confidence: {confidence:.4f}")

            return prediction

        except Exception as e:
            error_msg = f"LSTM prediction failed: {str(e)}"
            logger.error(error_msg)
            self.model_errors.append({'timestamp': datetime.now(), 'error': error_msg,
                'type': 'prediction_error'})
            return None

    def _fallback_strategy(self, df: pd.DataFrame) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Enhanced fallback strategy using Bollinger Bands + RSI.
        """
        logger.info("Using enhanced fallback strategy (Bollinger + RSI + Volume)")

        try:
            # Bollinger Bands
            bb_window = 20
            bb_std = 2.0
            rolling_mean = df['close'].rolling(window=bb_window).mean()
            rolling_std = df['close'].rolling(window=bb_window).std()
            upper_band = rolling_mean + (rolling_std * bb_std)
            middle_band = rolling_mean
            lower_band = rolling_mean - (rolling_std * bb_std)

            # RSI
            rsi = vbt.RSI.run(df['close'], window=14).rsi

            # Volume (or range proxy)
            if 'tick_volume' in df.columns:
                volume_ma = df['tick_volume'].rolling(20).mean()
                volume_filter = df['tick_volume'] > volume_ma * 1.2
            else:
                range_ma = (df['high'] - df['low']).rolling(20).mean()
                volume_filter = (df['high'] - df['low']) > range_ma * 1.2

            # Entry conditions
            price_above_middle = df['close'] > middle_band
            rsi_oversold_recovery = (rsi > 35) & (rsi < 70)
            not_overextended = df['close'] < (upper_band * 0.95)

            # Combine filters
            entries = (
                        price_above_middle & rsi_oversold_recovery & not_overextended & volume_filter)

            entries = entries.fillna(False).astype(int)

            # Stop loss and take profit
            sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
            tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

            logger.info(f"Fallback strategy generated {entries.sum()} signals")
            return entries, sl_stop, tp_stop

        except Exception as e:
            logger.error(f"Fallback strategy failed: {str(e)}")
            # Ultimate fallback - no signals
            return (pd.Series(0, index=df.index),
                    pd.Series(self.sl_fixed_percent, index=df.index),
                    pd.Series(self.tp_fixed_percent, index=df.index))

    def generate_signals(self, df: pd.DataFrame,
                         current_capital: Optional[float] = None) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Generate trading signals with LSTM + Order Block or fallback strategy.
        """
        logger.info(
            f"Generating signals for {self.symbol} using {'LSTM' if self.model_loaded else 'Fallback'} mode")

        try:
            # Always try order block detection first
            order_block_signals = self.detect_order_block(df)

            if self.model_loaded and not self.fallback_mode:
                # Try LSTM enhancement
                lstm_prediction = self._get_lstm_prediction(df)

                if lstm_prediction is not None:
                    # Combine LSTM with order blocks
                    lstm_bullish = lstm_prediction > self.lstm_threshold

                    # Enhanced signals: order blocks confirmed by LSTM
                    if lstm_bullish:
                        entries = order_block_signals
                        logger.info(
                            f"LSTM confirms bullish sentiment ({lstm_prediction:.3f} > {self.lstm_threshold})")
                    else:
                        entries = pd.Series(0,
                                            index=df.index)  # No signals if LSTM bearish
                        logger.info(
                            f"LSTM suggests bearish sentiment ({lstm_prediction:.3f} ≤ {self.lstm_threshold})")

                    # Stop loss and take profit
                    sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
                    tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

                    logger.info(
                        f"LSTM + Order Block strategy: {entries.sum()} final signals")
                    return entries, sl_stop, tp_stop
                else:
                    logger.warning(
                        "LSTM prediction failed, switching to order block only")
                    # Use order block signals only
                    entries = order_block_signals
                    sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
                    tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)
                    return entries, sl_stop, tp_stop
            else:
                # Use fallback strategy
                return self._fallback_strategy(df)

        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            # Ultimate fallback
            return self._fallback_strategy(df)

    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status information."""
        return {'tensorflow_available': TENSORFLOW_AVAILABLE,
            'model_loaded': self.model_loaded, 'model_validated': self.model_validated,
            'fallback_mode': self.fallback_mode,
            'recent_errors': len(self.model_errors),
            'last_model_check': self.last_model_check, 'avg_confidence': np.mean(
                self.model_confidence_history) if self.model_confidence_history else 0,
            'model_errors': self.model_errors[-5:] if self.model_errors else []
            # Last 5 errors
        }

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, Any]:
        """Get default parameters with error handling."""
        try:
            base_config = get_strategy_config("OrderBlockLSTMStrategy", timeframe)

            return {'symbol': [base_config.get('symbol', "GER40.cash")],
                'timeframe': [timeframe], 'ob_lookback': [15, 20, 25],
                'ob_strength': [1.5, 2.0, 2.5], 'lstm_threshold': [0.3, 0.4, 0.5, 0.6],
                'sl_fixed_percent': [0.015, 0.02, 0.025],
                'tp_fixed_percent': [0.03, 0.04, 0.05], 'use_trailing_stop': [True],
                'trailing_stop_percent': [0.008, 0.01, 0.015],
                'risk_per_trade': [0.005, 0.0075, 0.01],
                'model_confidence_threshold': [0.5, 0.6, 0.7]}
        except Exception as e:
            logger.error(f"Error getting default params: {e}")
            # Fallback params
            return {'symbol': ["GER40.cash"], 'ob_lookback': [20],
                'sl_fixed_percent': [0.02], 'tp_fixed_percent': [0.04]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """Get parameter descriptions."""
        return {'symbol': 'Trading symbol for the strategy',
            'timeframe': 'Chart timeframe for analysis',
            'ob_lookback': 'Periods to analyze for order block detection',
            'ob_strength': 'Minimum strength multiplier for order block validation',
            'lstm_threshold': 'LSTM prediction threshold for bullish signals',
            'sl_fixed_percent': 'Stop-loss percentage',
            'tp_fixed_percent': 'Take-profit percentage',
            'use_trailing_stop': 'Enable trailing stop functionality',
            'trailing_stop_percent': 'Trailing stop percentage',
            'risk_per_trade': 'Risk per trade as percentage of capital',
            'model_confidence_threshold': 'Minimum confidence for LSTM predictions'}