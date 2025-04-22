import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from .base_strategy import BaseStrategy


class AILSTMStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.model = load_model(
            'lstm_trading_model.h5')  # Laad het getrainde LSTM-model
        self.seq_length = 50  # Sequentielengte voor het LSTM-model
        self.buy_threshold = 0.3  # Drempel voor buy-signaal
        self.sell_threshold = -0.3  # Drempel voor sell-signaal

    def generate_signals(self, df: pd.DataFrame) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Genereer buy-, sell- en hold-signalen op basis van het LSTM-model.

        Args:
            df: DataFrame met historische OHLC-data.

        Returns:
            Tuple van (buy_signals, sell_signals, hold_signals) als pandas Series.
        """
        N = len(df)
        predictions = pd.Series(index=df.index, dtype=float)

        # Genereer voorspellingen voor elk tijdstip t >= seq_length
        for t in range(self.seq_length, N):
            X_t = df.iloc[t - self.seq_length: t][
                ['open', 'high', 'low', 'close']].values
            X_t = np.expand_dims(X_t,
                                 axis=0)  # Voeg batch-dimensie toe: (1, seq_length, 4)
            pred = self.model.predict(X_t, verbose=0)[0][0]  # Haal de voorspelling op
            predictions.iloc[t] = pred

        # Genereer signalen op basis van de voorspellingen
        buy_signals = (predictions > self.buy_threshold).astype(int)
        sell_signals = (predictions < self.sell_threshold).astype(int)
        hold_signals = pd.Series(1 - (buy_signals + sell_signals), index=df.index)

        return buy_signals, sell_signals, hold_signals