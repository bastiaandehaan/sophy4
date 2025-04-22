# strategies/order_block_lstm_strategy.py
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Controleer of TensorFlow geïnstalleerd is
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow")
    sys.exit(1)

from strategies.base_strategy import BaseStrategy
from strategies import register_strategy
from config import logger


class OrderBlock:
    """Representeert een order block zone in de markt."""

    def __init__(self, direction: int, time, high: float, low: float):
        self.direction = direction  # 1 = bullish, -1 = bearish
        self.time = time
        self.high = high
        self.low = low
        self.traded = False


@register_strategy
class OrderBlockLSTMStrategy(BaseStrategy):
    """
    Trading strategie gebaseerd op order blocks, Fibonacci retracements en LSTM-voorspellingen.
    """

    def __init__(self, window: int = 60, lstm_threshold: float = 0.0,
                 sl_fixed_percent: float = 0.01, tp_fixed_percent: float = 0.02,
                 use_trailing_stop: bool = False, trailing_stop_percent: float = 0.015,
                 risk_per_trade: float = 0.01, confidence_level: float = 0.95,
                 model_path: Optional[str] = None):
        """
        Initialiseer de OrderBlockLSTMStrategy.
        """
        super().__init__()
        self.window = window
        self.lstm_threshold = lstm_threshold
        self.sl_fixed_percent = sl_fixed_percent
        self.tp_fixed_percent = tp_fixed_percent
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_percent = trailing_stop_percent
        self.risk_per_trade = risk_per_trade
        self.confidence_level = confidence_level

        # Laad model, indien beschikbaar
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                logger.info(f"LSTM model geladen van {model_path}")
            except Exception as e:
                logger.error(f"Kan LSTM model niet laden: {str(e)}")
        else:
            logger.warning(
                "Geen LSTM model opgegeven of gevonden, gebruik fallback logica")

    def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detecteer order blocks in de data.
        """
        obs = []
        for i in range(len(df) - 3):
            c1, c2, c3 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2]

            # Bearish->Bearish->Bullish engulf = Bullish OB
            if (c1['open'] > c1['close'] and c2['open'] > c2['close'] and c3['open'] <
                    c3['close'] and c3['open'] < c2['close']):
                obs.append(OrderBlock(1, c3.name, c3['high'], c3['low']))

            # Bullish->Bullish->Bearish engulf = Bearish OB
            elif (c1['open'] < c1['close'] and c2['open'] < c2['close'] and c3['open'] >
                  c3['close'] and c3['open'] > c2['close']):
                obs.append(OrderBlock(-1, c3.name, c3['high'], c3['low']))

        return obs

    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float) -> Dict[
        str, float]:
        """
        Bereken Fibonacci retracement levels.
        """
        diff = swing_high - swing_low
        return {'61.8%': swing_high - 0.618 * diff, '50.0%': swing_high - 0.5 * diff,
            '38.2%': swing_high - 0.382 * diff, }

    def prepare_lstm_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Bereid input voor voor het LSTM model.
        """
        if len(df) < self.window:
            logger.warning(
                f"Te weinig data voor LSTM (nodig: {self.window}, beschikbaar: {len(df)})")
            return np.zeros((1, self.window, 2))  # Fallback empty input

        # Extract the most recent window of close and volume data
        recent_data = df.iloc[-self.window:]
        seq = [[row['close'], row.get('volume', 0)] for _, row in
               recent_data.iterrows()]
        return np.array(seq).reshape(1, self.window, 2)

    def generate_signals(self, df: pd.DataFrame) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Genereer entries, sl_stop, tp_stop op basis van order blocks, Fibonacci en LSTM.
        """
        # Initialiseer lege Series met False/0
        entries = pd.Series(False, index=df.index)
        sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
        tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

        # Detecteer order blocks
        order_blocks = self.detect_order_blocks(df)
        if not order_blocks:
            logger.info("Geen order blocks gedetecteerd")
            return entries, sl_stop, tp_stop

        # Bereid LSTM-voorspelling voor als model beschikbaar is
        lstm_pred = 0.0
        if self.model is not None:
            X_in = self.prepare_lstm_input(df)
            try:
                lstm_pred = self.model.predict(X_in, verbose=0)[0][0]
                logger.info(f"LSTM predictie: {lstm_pred:.4f}")
            except Exception as e:
                logger.error(f"LSTM predictie mislukt: {str(e)}")

        # Loop door order blocks om signalen te genereren
        current_price = df['close'].iloc[-1]

        for ob in order_blocks:
            # Bereken Fibonacci niveaus
            past_data = df.loc[:ob.time]
            if past_data.empty:
                continue

            swing_high = past_data['high'].max()
            swing_low = past_data['low'].min()
            fib = self.calculate_fibonacci_levels(swing_high, swing_low)

            # Genereer signaal op de meest recente candle als aan voorwaarden wordt voldaan
            if ob.direction == 1 and fib[
                '61.8%'] <= current_price <= ob.high and lstm_pred > self.lstm_threshold:
                # Bullish signaal
                entries.iloc[-1] = True
                logger.info(f"Bullish signaal gegenereerd op {df.index[-1]}")
            elif ob.direction == -1 and ob.low <= current_price <= fib[
                '61.8%'] and lstm_pred < -self.lstm_threshold:
                # Bearish signaal (niet ondersteund in deze implementatie)
                logger.info(
                    f"Bearish signaal gedetecteerd maar alleen long posities worden ondersteund")

        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """
        Geef default parameters voor de strategie.
        """
        return {'window': [30, 50, 60], 'lstm_threshold': [0.0, 0.1, 0.2],
            'sl_fixed_percent': [0.01, 0.015, 0.02],
            'tp_fixed_percent': [0.02, 0.03, 0.04], 'use_trailing_stop': [True, False],
            'trailing_stop_percent': [0.01, 0.015, 0.02],
            'risk_per_trade': [0.005, 0.01, 0.02],
            'confidence_level': [0.90, 0.95, 0.99]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """
        Beschrijf parameters voor documentatie.
        """
        return {'window': 'Aantal perioden voor LSTM input sequentie',
            'lstm_threshold': 'Drempelwaarde voor LSTM signalen (0 tot 1)',
            'sl_fixed_percent': 'Stop-loss als vast percentage',
            'tp_fixed_percent': 'Take-profit als vast percentage',
            'use_trailing_stop': 'Trailing stop activeren (true/false)',
            'trailing_stop_percent': 'Trailing stop als vast percentage',
            'risk_per_trade': 'Risico per trade als percentage van portfolio (0.01 = 1%)',
            'confidence_level': 'Betrouwbaarheidsniveau voor VaR-berekening',
            'model_path': 'Pad naar voorgetraind LSTM-model (.h5 bestand)'}