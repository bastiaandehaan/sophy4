import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from strategies import register_strategy
from strategies.base_strategy import BaseStrategy
from risk.risk_management import RiskManager

logger = logging.getLogger(__name__)


# Je bestaande AttentionLayer en andere hulpfuncties blijven hier...

@register_strategy
class OrderBlockLSTMStrategy(BaseStrategy):
    """
    Advanced strategy combining Order Block detection with LSTM predictions.
    """

    def __init__(self, symbol: str = "EURUSD", model_path: Optional[str] = None,
                 seq_len: int = 50, lstm_threshold: float = 0.3, ob_lookback: int = 20,
                 ob_strength: float = 1.5, sl_fixed_percent: float = 0.02,
                 tp_fixed_percent: float = 0.03, use_trailing_stop: bool = True,
                 trailing_stop_percent: float = 0.015, risk_per_trade: float = 0.01,
                 confidence_level: float = 0.95):
        """
        Initialize the OrderBlockLSTMStrategy.

        Args:
            symbol: Trading symbol
            model_path: Path to trained LSTM model
            seq_len: Sequence length for LSTM
            lstm_threshold: Threshold for LSTM predictions
            ob_lookback: Lookback period for order blocks
            ob_strength: Minimum strength for order blocks
            sl_fixed_percent: Fixed stop-loss percentage
            tp_fixed_percent: Fixed take-profit percentage
            use_trailing_stop: Whether to use trailing stop
            trailing_stop_percent: Trailing stop percentage
            risk_per_trade: Risk per trade as portfolio percentage
            confidence_level: Confidence level for VaR
        """
        super().__init__()
        self.symbol = symbol
        self.model_path = model_path
        self.seq_len = seq_len
        self.lstm_threshold = lstm_threshold
        self.ob_lookback = ob_lookback
        self.ob_strength = ob_strength
        self.sl_fixed_percent = sl_fixed_percent
        self.tp_fixed_percent = tp_fixed_percent
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_percent = trailing_stop_percent
        self.risk_per_trade = risk_per_trade
        self.confidence_level = confidence_level

        # Load LSTM model if path provided
        self.lstm_model = None
        if model_path and Path(model_path).exists():
            try:
                self.lstm_model = load_model(model_path)
                logger.info(f"LSTM model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}")

    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.seq_len < 10:
            raise ValueError("seq_len must be at least 10")
        if not 0 < self.lstm_threshold < 1:
            raise ValueError("lstm_threshold must be between 0 and 1")
        if self.ob_lookback < 5:
            raise ValueError("ob_lookback must be at least 5")
        if self.ob_strength <= 0:
            raise ValueError("ob_strength must be positive")
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            raise ValueError("risk_per_trade must be between 0 and 0.1")
        return True

    def detect_order_blocks(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect order blocks in price action.

        Returns:
            Tuple of (bullish_ob, bearish_ob) boolean series
        """
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['tick_volume']

        # Calculate average volume
        avg_volume = volume.rolling(window=self.ob_lookback).mean()

        # Detect bullish order blocks (strong buying)
        bullish_ob = ((close > close.shift(1)) &  # Price up
                      (volume > avg_volume * self.ob_strength) &  # High volume
                      (close > high.shift(1))  # Breakout
        )

        # Detect bearish order blocks (strong selling)
        bearish_ob = ((close < close.shift(1)) &  # Price down
                      (volume > avg_volume * self.ob_strength) &  # High volume
                      (close < low.shift(1))  # Breakdown
        )

        return bullish_ob, bearish_ob

    def prepare_lstm_data(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare data for LSTM prediction."""
        if self.lstm_model is None:
            return None

        try:
            from sklearn.preprocessing import MinMaxScaler
            # Prepare features (similar to training)
            features = ['open', 'high', 'low', 'close', 'tick_volume']
            feature_data = df[features].values

            # Normalize
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(feature_data)

            # Create sequences
            sequences = []
            for i in range(len(scaled_data) - self.seq_len + 1):
                sequences.append(scaled_data[i:i + self.seq_len])

            return np.array(sequences)
        except Exception as e:
            logger.error(f"Failed to prepare LSTM data: {e}")
            return None

    def generate_signals(self, df: pd.DataFrame,
                         current_capital: Optional[float] = None) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """Generate trading signals combining order blocks and LSTM predictions."""
        # Initialize empty signals
        entries = pd.Series(0, index=df.index)
        sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
        tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

        # Detect order blocks
        bullish_ob, bearish_ob = self.detect_order_blocks(df)

        # LSTM predictions (if model available)
        lstm_bullish = pd.Series(False, index=df.index)
        lstm_bearish = pd.Series(False, index=df.index)

        if self.lstm_model is not None:
            sequences = self.prepare_lstm_data(df)
            if sequences is not None and len(sequences) > 0:
                try:
                    predictions = self.lstm_model.predict(sequences, verbose=0)

                    # Align predictions with dataframe index
                    pred_index = df.index[self.seq_len - 1:]
                    pred_series = pd.Series(predictions.flatten(), index=pred_index)

                    # Create LSTM signals
                    lstm_bullish.loc[pred_index] = pred_series > self.lstm_threshold
                    lstm_bearish.loc[pred_index] = pred_series < -self.lstm_threshold
                except Exception as e:
                    logger.error(f"LSTM prediction failed: {e}")

        # Combine signals
        if self.lstm_model is not None:
            # Both OB and LSTM must agree
            entries.loc[bullish_ob & lstm_bullish] = 1
            entries.loc[bearish_ob & lstm_bearish] = -1
        else:
            # Only use order blocks
            entries.loc[bullish_ob] = 1
            entries.loc[bearish_ob] = -1

        # Risk management
        risk_manager = RiskManager(confidence_level=self.confidence_level,
            max_risk=self.risk_per_trade)

        # Trailing stop logic
        if self.use_trailing_stop:
            sl_stop = sl_stop.fillna(self.trailing_stop_percent)

        logger.info(f"Generated {entries.abs().sum()} signals")
        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """Get default parameters for optimization."""
        return {'symbol': ["EURUSD", "GBPUSD", "XAUUSD"], 'seq_len': [30, 50, 70],
            'lstm_threshold': [0.2, 0.3, 0.4], 'ob_lookback': [15, 20, 25],
            'ob_strength': [1.2, 1.5, 1.8], 'sl_fixed_percent': [0.01, 0.015, 0.02],
            'tp_fixed_percent': [0.02, 0.03, 0.04], 'use_trailing_stop': [True],
            'trailing_stop_percent': [0.01, 0.015, 0.02],
            'risk_per_trade': [0.005, 0.01, 0.015],
            'confidence_level': [0.90, 0.95, 0.99]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """Get parameter descriptions."""
        return {'symbol': 'Trading symbol (e.g., EURUSD, XAUUSD)',
            'model_path': 'Path to trained LSTM model file',
            'seq_len': 'LSTM sequence length for predictions',
            'lstm_threshold': 'Threshold for LSTM buy/sell signals',
            'ob_lookback': 'Lookback period for order block detection',
            'ob_strength': 'Minimum volume multiplier for order blocks',
            'sl_fixed_percent': 'Fixed stop-loss percentage',
            'tp_fixed_percent': 'Fixed take-profit percentage',
            'use_trailing_stop': 'Enable trailing stop-loss',
            'trailing_stop_percent': 'Trailing stop percentage',
            'risk_per_trade': 'Risk per trade as portfolio percentage',
            'confidence_level': 'Confidence level for VaR calculation'}

# Behoud je bestaande training functies...