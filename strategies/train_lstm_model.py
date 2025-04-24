import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow")
    Sequential = None
    LSTM = None
    Dropout = None
    load_model = None
    Adam = None
    exit(1)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import logger
from strategies.base_strategy import BaseStrategy



def prepare_data(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare data for LSTM training by creating sequences.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data.
        seq_len (int): Length of the sequence for LSTM input.

    Returns:
        Tuple[np.ndarray, np.ndarray, MinMaxScaler]: (X, y, scaler) where X is the input sequences, y is the target, and scaler is the fitted scaler.
    """
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame missing required columns: {required_columns}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close']])

    X, y = [], []
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i:i + seq_len])
        y.append(scaled_data[i + seq_len, 3])  # Predict the 'close' price

    return np.array(X), np.array(y), scaler


def build_lstm_model(seq_len: int, n_features: int = 4) -> Sequential:
    """
    Build and compile an LSTM model.

    Args:
        seq_len (int): Length of the sequence.
        n_features (int): Number of features (default 4 for OHLC).

    Returns:
        Sequential: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_and_save_model(
    symbol: str,
    timeframe: str,
    days: int,
    seq_len: int,
    output_dir: str,
    verbose: int = 1
) -> None:
    """
    Train an LSTM model for a given symbol and timeframe and save it.

    Args:
        symbol (str): Trading symbol (e.g., 'XAUUSD').
        timeframe (str): Timeframe (e.g., 'H1').
        days (int): Number of days of historical data.
        seq_len (int): Sequence length for LSTM.
        output_dir (str): Directory to save the trained model.
        verbose (int): Verbosity level for training (0 = silent, 1 = progress bar).
    """
    logger.info(f"Starting LSTM training for {symbol} on {timeframe}")

    # Fetch historical data
    df = BaseStrategy.fetch_historical_data(symbol=symbol, timeframe=timeframe, days=days)
    if df is None or df.empty:
        logger.error(f"Failed to fetch historical data for {symbol} on {timeframe}")
        return

    logger.info(f"Fetched {len(df)} candles for {symbol} on {timeframe}")

    # Prepare data
    X, y, scaler = prepare_data(df, seq_len)
    if len(X) == 0:
        logger.error(f"Insufficient data to train LSTM for {symbol} on {timeframe}")
        return

    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train model
    model = build_lstm_model(seq_len)
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=verbose
    )

    # Save the model
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    model_path = output_path / f"lstm_{symbol}_{timeframe}.h5"
    model.save(str(model_path))
    logger.info(f"LSTM model saved to {model_path}")


def main() -> None:
    """Main function to train LSTM model from command line arguments."""
    parser = argparse.ArgumentParser(description="Train LSTM model for trading strategy")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., XAUUSD)")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (e.g., H1)")
    parser.add_argument("--days", type=int, default=500, help="Number of days of historical data")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for LSTM")
    parser.add_argument("--output", type=str, default="models", help="Output directory for the model")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0 = silent, 1 = progress bar)")
    args = parser.parse_args()

    train_and_save_model(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        seq_len=args.seq_len,
        output_dir=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()