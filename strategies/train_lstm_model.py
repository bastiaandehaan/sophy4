# strategies/train_lstm_model.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# TensorFlow imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow")
    sys.exit(1)

# Voeg projectroot toe aan Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import fetch_historical_data
from config import logger


def create_sequences(df: pd.DataFrame, seq_len: int = 60) -> tuple:
    """
    Maak sequenties voor LSTM training van OHLCV data.
    """
    X, y = [], []
    # Bereken prijsveranderingen als target
    df['target'] = df['close'].pct_change(5).shift(-5)  # 5-bar toekomstige verandering

    # Loop door de data om sequenties te maken
    for i in range(len(df) - seq_len - 5):
        # Feature sequentie
        seq = df.iloc[i:i + seq_len][['close', 'volume']].values
        X.append(seq)

        # Target (tussen -1 en 1 met tanh)
        target = df.iloc[i + seq_len]['target']
        # Schaal tussen -1 en 1
        target = np.tanh(target * 10)  # *10 voor betere schaling
        y.append(target)

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
        return

    logger.info(f"Data geladen: {len(df)} rijen")

    # Maak sequenties
    X, y = create_sequences(df, seq_len)
    logger.info(f"Sequenties gemaakt: {len(X)} samples")

    if len(X) < 100:
        logger.error("Te weinig data voor model training")
        return

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

    print(f"Model succesvol getraind en opgeslagen: {model_path}")