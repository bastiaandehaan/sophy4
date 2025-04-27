import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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


def prepare_data(df: pd.DataFrame, seq_len: int, target_column: str = 'close',
                 target_shift: int = 1, feature_columns: list = None) -> Tuple[
    np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare data for LSTM training by creating sequences with improved feature selection.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data.
        seq_len (int): Length of the sequence for LSTM input.
        target_column (str): Column to predict (default: 'close').
        target_shift (int): How many periods ahead to predict (default: 1).
        feature_columns (list): Columns to use as features (default: OHLC).

    Returns:
        Tuple[np.ndarray, np.ndarray, MinMaxScaler]: (X, y, scaler)
    """
    if feature_columns is None:
        feature_columns = ['open', 'high', 'low', 'close']

    required_columns = feature_columns + [
        target_column] if target_column not in feature_columns else feature_columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame missing required columns: {required_columns}")

    # Create technical indicators
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)

    # Add features to feature_columns if they're not already there
    enhanced_features = feature_columns.copy()
    for indicator in ['ma20', 'ma50', 'rsi']:
        if indicator not in enhanced_features:
            enhanced_features.append(indicator)

    # Fill missing values from indicators
    df = df.fillna(method='bfill')

    # Normalize features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[enhanced_features])

    # Create shifted target (e.g., predict close price 1 period ahead)
    target_idx = enhanced_features.index(target_column)
    shifted_target = df[target_column].shift(-target_shift).dropna().values

    # Adjust data length to match target length
    data_length = len(shifted_target)
    scaled_data = scaled_data[:data_length]

    X, y = [], []
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i:i + seq_len])
        y.append(shifted_target[i + seq_len])

    return np.array(X), np.array(y), scaler


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def build_lstm_model(seq_len: int, n_features: int = 4,
                     lstm_units: int = 50) -> Sequential:
    """
    Build and compile an improved LSTM model.

    Args:
        seq_len (int): Length of the sequence.
        n_features (int): Number of features.
        lstm_units (int): Size of LSTM layers.

    Returns:
        Sequential: Compiled LSTM model.
    """
    model = Sequential(
        [LSTM(lstm_units, return_sequences=True, input_shape=(seq_len, n_features)),
            Dropout(0.2), LSTM(lstm_units, return_sequences=False), Dropout(0.2),
            Dense(25, activation='relu'), Dense(1)])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, scaler,
                   feature_columns: list, results_dir: str, symbol: str) -> Dict[
    str, float]:
    """
    Evaluate model and create performance visualizations.

    Returns:
        Dict with evaluation metrics.
    """
    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))

    # Calculate correlation coefficient
    corr = np.corrcoef(predictions.flatten(), y_test)[0, 1]

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'LSTM Performance for {symbol}')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'lstm_performance_{symbol}.png'))

    # Calculate directional accuracy
    direction_actual = np.diff(y_test)
    direction_pred = np.diff(predictions.flatten())
    directional_accuracy = np.mean((direction_actual > 0) == (direction_pred > 0))

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'correlation': corr,
        'directional_accuracy': directional_accuracy}


def train_and_save_model(symbol: str, timeframe: str, days: int, seq_len: int,
        output_dir: str, lstm_units: int = 50, epochs: int = 50, batch_size: int = 32,
        verbose: int = 1) -> Dict[str, Any]:
    """
    Train an LSTM model for a given symbol and timeframe with improved visualization.

    Args:
        symbol (str): Trading symbol (e.g., 'XAUUSD').
        timeframe (str): Timeframe (e.g., 'H1').
        days (int): Number of days of historical data.
        seq_len (int): Sequence length for LSTM.
        output_dir (str): Directory to save the trained model.
        lstm_units (int): Number of LSTM units per layer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        verbose (int): Verbosity level for training.

    Returns:
        Dict with training results.
    """
    logger.info(f"Starting LSTM training for {symbol} on {timeframe}")

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    model_dir = output_path / "models"
    model_dir.mkdir(exist_ok=True)

    results_dir = output_path / "results"
    results_dir.mkdir(exist_ok=True)

    # Fetch historical data
    df = BaseStrategy.fetch_historical_data(symbol=symbol, timeframe=timeframe,
                                            days=days)
    if df is None or df.empty:
        logger.error(f"Failed to fetch historical data for {symbol} on {timeframe}")
        return {"success": False, "error": "Failed to fetch data"}

    logger.info(f"Fetched {len(df)} candles for {symbol} on {timeframe}")

    # Enhanced features for better predictions
    feature_columns = ['open', 'high', 'low', 'close', 'tick_volume']
    if 'spread' in df.columns:
        feature_columns.append('spread')

    # Prepare data with enhanced features
    try:
        X, y, scaler = prepare_data(df, seq_len, feature_columns=feature_columns)
        if len(X) == 0:
            logger.error(f"Insufficient data to train LSTM for {symbol} on {timeframe}")
            return {"success": False, "error": "Insufficient data"}
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        return {"success": False, "error": f"Data preparation error: {str(e)}"}

    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build model with specified units
    n_features = X.shape[2]
    model = build_lstm_model(seq_len, n_features, lstm_units)

    # Setup callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(model_dir / f"lstm_{symbol}_{timeframe}_checkpoint.h5"),
            save_best_only=True, monitor='val_loss')]

    # Train model with callbacks
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test), callbacks=callbacks, verbose=verbose)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'LSTM Training History for {symbol}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(results_dir / f"lstm_training_{symbol}_{timeframe}.png"))

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, scaler, feature_columns,
                             str(results_dir), symbol)
    logger.info(f"Model evaluation metrics: {metrics}")

    # Save final model
    model_path = model_dir / f"lstm_{symbol}_{timeframe}.h5"
    model.save(str(model_path))
    logger.info(f"LSTM model saved to {model_path}")

    # Save a text file with results summary
    with open(str(results_dir / f"results_{symbol}_{timeframe}.txt"), 'w') as f:
        f.write(f"LSTM Training Results for {symbol} on {timeframe}\n")
        f.write(f"Sequence Length: {seq_len}\n")
        f.write(f"LSTM Units: {lstm_units}\n")
        f.write(f"Data Points: {len(df)}\n")
        f.write(f"Features: {feature_columns}\n\n")
        f.write("Evaluation Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")

    return {"success": True, "model_path": str(model_path), "metrics": metrics,
        "data_points": len(df), "sequence_length": seq_len}


def main() -> None:
    """Main function to train LSTM model from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LSTM model for trading strategy")
    parser.add_argument("--symbol", type=str, required=True,
                        help="Trading symbol (e.g., XAUUSD)")
    parser.add_argument("--timeframe", type=str, default="H1",
                        help="Timeframe (e.g., H1)")
    parser.add_argument("--days", type=int, default=500,
                        help="Number of days of historical data")
    parser.add_argument("--seq_len", type=int, default=50,
                        help="Sequence length for LSTM")
    parser.add_argument("--output", type=str, default="models",
                        help="Output directory for the model")
    parser.add_argument("--lstm_units", type=int, default=50,
                        help="Number of LSTM units per layer")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=silent, 1=progress, 2=detailed)")
    args = parser.parse_args()

    train_and_save_model(symbol=args.symbol, timeframe=args.timeframe, days=args.days,
        seq_len=args.seq_len, output_dir=args.output, lstm_units=args.lstm_units,
        epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose)


if __name__ == "__main__":
    main()