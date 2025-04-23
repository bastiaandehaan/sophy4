# strategies/train_lstm_model.py
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# TensorFlow imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow")
    sys.exit(1)

# Voeg projectroot toe aan Python path
sys.path.append(str(Path(__file__).parent.parent))

from backtest.data_loader import fetch_historical_data
from config import logger


def create_sequences(df: pd.DataFrame, seq_len: int = 60) -> tuple:
    """
    Maak sequenties voor LSTM training van OHLCV data.

    Args:
        df: DataFrame met OHLC data
        seq_len: Sequentielengte voor LSTM

    Returns:
        Tuple van (X, y) data
    """
    X, y = [], []

    # Controleer welke volume kolom beschikbaar is
    volume_column = None
    for possible_name in ['volume', 'tick_volume', 'real_volume']:
        if possible_name in df.columns:
            volume_column = possible_name
            logger.info(f"Volume kolom gevonden: '{volume_column}'")
            break

    # Als geen volume kolom gevonden, maak een dummy kolom met waarde 1
    if volume_column is None:
        logger.warning("Geen volume kolom gevonden, gebruik dummy waarden")
        df['volume_dummy'] = 1.0
        volume_column = 'volume_dummy'

    # Normaliseer volume (om numerieke stabiliteitsproblemen te voorkomen)
    df[f'{volume_column}_norm'] = df[volume_column] / df[volume_column].rolling(
        window=20).mean().fillna(1)

    # Bereken prijsveranderingen als target
    df['target'] = df['close'].pct_change(5).shift(-5)  # 5-bar toekomstige verandering

    # Log de eerste paar rijen om te debuggen
    logger.info(f"DataFrame voor sequence creation:\n{df.head().to_string()}")

    # Controleer of er voldoende data is na NaN verwijdering
    df_clean = df.dropna()
    if len(df_clean) < seq_len + 5:
        logger.warning(f"Te weinig data na NaN verwijdering: {len(df_clean)} rijen")
        return np.array([]), np.array([])

    # Loop door de data om sequenties te maken
    for i in range(len(df_clean) - seq_len - 5):
        # Feature sequentie (gebruik genormaliseerde volume)
        seq = df_clean.iloc[i:i + seq_len][['close', f'{volume_column}_norm']].values
        X.append(seq)

        # Target (tussen -1 en 1 met tanh)
        target = df_clean.iloc[i + seq_len]['target']
        # Schaal tussen -1 en 1
        target = np.tanh(target * 10)  # *10 voor betere schaling
        y.append(target)

    logger.info(f"Aantal sequenties gecreëerd: {len(X)}")
    return np.array(X), np.array(y)


def train_and_save_model(symbol: str, timeframe: str = "H1", days: int = 500,
                         seq_len: int = 60, output: str = "models"):
    """
    Train LSTM model en sla het op.
    """
    logger.info(f"Ophalen data voor {symbol} ({timeframe}, {days} dagen)...")
    df = fetch_historical_data(symbol, timeframe=timeframe, days=days)

    if df is None or df.empty:
        logger.error(f"Geen data ontvangen voor {symbol}")
        return None

    logger.info(f"Data geladen: {len(df)} rijen, kolommen: {list(df.columns)}")

    # Maak sequenties
    X, y = create_sequences(df, seq_len)
    if len(X) == 0:
        logger.error("Geen sequenties konden worden gemaakt")
        return None

    logger.info(f"Sequenties gemaakt: {len(X)} samples, shape: {X.shape}")

    if len(X) < 100:
        logger.warning("Weinig data voor model training (<100 samples)")
        if len(X) < 20:
            logger.error("Te weinig data voor model training")
            return None

    # Train-test split (80-20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Modelarchitectuur
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1, activation='tanh'))  # tanh voor waarden tussen -1 en 1

    # Compileer
    model.compile(optimizer='adam', loss='mse')

    # Train
    logger.info("Start training...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20,
              batch_size=32, verbose=1)

    # Maak output directory
    Path(output).mkdir(parents=True, exist_ok=True)

    # Sla model op
    model_path = f"{output}/lstm_{symbol}_{timeframe}.h5"
    model.save(model_path)
    logger.info(f"Model opgeslagen: {model_path}")

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM model voor Sophy4 OrderBlockLSTM strategy")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Handelssymbool")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe")
    parser.add_argument("--days", type=int, default=500, help="Aantal dagen data")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequentielengte")
    parser.add_argument("--output", type=str, default="models", help="Output directory")

    args = parser.parse_args()

    model_path = train_and_save_model(args.symbol, args.timeframe, args.days,
                                      args.seq_len, args.output)

    if model_path:
        print(f"Model succesvol getraind en opgeslagen: {model_path}")
    else:
        print("Model training mislukt")