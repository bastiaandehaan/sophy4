import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime, timedelta
import time
import os
import psutil  # For resource monitoring

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
except ImportError:
    print("TensorFlow or Keras Tuner not installed. Please run: pip install tensorflow keras-tuner")
    exit(1)

import sys
import vectorbt as vbt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import logger
from strategies.base_strategy import BaseStrategy

# Custom callback for trial progress and resource monitoring
class TrialProgressCallback(Callback):
    def __init__(self, trial_num, total_trials):
        super().__init__()
        self.trial_num = trial_num
        self.total_trials = total_trials
        self.trial_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        # Monitor CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        print(f"   Trial {self.trial_num}/{self.total_trials} - Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"   CPU Usage: {cpu_usage:.1f}% | Memory Usage: {memory_usage:.1f}%")

# Attention Layer for LSTM
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Alignment scores
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        # Attention weights
        alpha = tf.keras.backend.softmax(e, axis=1)
        # Context vector
        context = inputs * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context

def prepare_data(df: pd.DataFrame, seq_len: int, target_column: str = 'close',
                 target_shift: int = 1, feature_columns: list = None) -> Tuple[
    np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare data for LSTM training with enhanced feature engineering.
    """
    if feature_columns is None:
        feature_columns = ['open', 'high', 'low', 'close']

    required_columns = feature_columns + [target_column] if target_column not in feature_columns else feature_columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame missing required columns: {required_columns}")

    # Enhanced Feature Engineering with VectorBT (matching OrderBlockLSTMStrategy)
    print("🔧 Calculating technical indicators...")
    df['ma20'] = vbt.MA.run(df['close'], window=20).ma
    df['ma50'] = vbt.MA.run(df['close'], window=50).ma
    df['rsi'] = vbt.RSI.run(df['close'], window=14).rsi
    df['atr'] = vbt.ATR.run(df['high'], df['low'], df['close'], window=14).atr
    bb = vbt.BBANDS.run(df['close'], window=20)
    df['bb_upper'] = bb.upper
    df['bb_lower'] = bb.lower
    macd = vbt.MACD.run(df['close'], fast_window=12, slow_window=26, signal_window=9)
    df['macd'] = macd.macd
    df['macd_signal'] = macd.signal

    # Orderblock-specific features (to match OrderBlockLSTMStrategy)
    df['price_movement'] = df['close'] - df['open']
    df['candle_range'] = df['high'] - df['low']
    df['volume_ma'] = df['tick_volume'].rolling(window=20).mean() if 'tick_volume' in df.columns else df['candle_range'].rolling(window=20).mean()
    df['volume_ratio'] = df['tick_volume'] / df['volume_ma'] if 'tick_volume' in df.columns else df['candle_range'] / df['candle_range'].rolling(window=20).mean()
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['sma200'] = vbt.MA.run(df['close'], window=200).ma
    df['trend'] = (df['close'] > df['sma200']).astype(float)

    # Add features to feature_columns
    enhanced_features = feature_columns.copy()
    for indicator in ['ma20', 'ma50', 'rsi', 'atr', 'bb_upper', 'bb_lower', 'macd',
                      'macd_signal', 'price_movement', 'candle_range', 'volume_ratio',
                      'price_position', 'trend']:
        if indicator not in enhanced_features:
            enhanced_features.append(indicator)

    # Fill missing values
    df = df.ffill()  # Fixed to use ffill() instead of deprecated method

    # Normalize features
    print("📊 Normalizing data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[enhanced_features])

    # Create shifted target
    target_idx = enhanced_features.index(target_column)
    shifted_target = df[target_column].shift(-target_shift).dropna().values

    # Adjust data length
    data_length = len(shifted_target)
    scaled_data = scaled_data[:data_length]

    X, y = [], []
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i:i + seq_len])
        y.append(shifted_target[i + seq_len])

    return np.array(X), np.array(y), scaler

def build_lstm_model(hp, seq_len: int, n_features: int) -> Sequential:
    """
    Build and compile an LSTM model with attention mechanism and hyperparameter tuning.
    """
    model = Sequential()
    model.add(Input(shape=(seq_len, n_features)))

    # Reduced LSTM layers for efficiency
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=64, step=16)  # Reduced range
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.3, step=0.1)))
    model.add(LSTM(units=lstm_units // 2, return_sequences=True))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.3, step=0.1)))

    # Attention mechanism
    model.add(AttentionLayer())

    # Dense layers
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=32, step=8),  # Reduced range
                    activation='relu'))
    model.add(Dense(1))

    # Compile with tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='LOG')  # Reduced range
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, scaler,
                   feature_columns: list, results_dir: str, symbol: str) -> Dict[str, float]:
    """
    Evaluate model and create performance visualizations.
    """
    print("📈 Evaluating model performance...")
    predictions = model.predict(X_test, verbose=0)

    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    corr = np.corrcoef(predictions.flatten(), y_test)[0, 1]

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'LSTM Performance for {symbol}')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'lstm_performance_{symbol}.png'))
    plt.close()

    # Directional accuracy
    direction_actual = np.diff(y_test)
    direction_pred = np.diff(predictions.flatten())
    directional_accuracy = np.mean((direction_actual > 0) == (direction_pred > 0))

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'correlation': corr,
            'directional_accuracy': directional_accuracy}

def train_and_save_model(symbol: str, timeframe: str, days: int, seq_len: int,
                         output_dir: str, epochs: int = 30, batch_size: int = 64,  # Adjusted defaults
                         verbose: int = 1) -> Dict[str, Any]:
    """
    Train an LSTM model with hyperparameter tuning and attention mechanism.
    """
    training_start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"🚀 LSTM MODEL TRAINING")
    print(f"{'=' * 60}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {days} days | {seq_len} sequence | {epochs} epochs")
    print(f"{'=' * 60}\n")

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    model_dir = output_path / "trainedh5"
    model_dir.mkdir(exist_ok=True)

    results_dir = output_path / "results"
    results_dir.mkdir(exist_ok=True)

    # Fetch historical data
    print("📥 Fetching historical data...")
    df = BaseStrategy.fetch_historical_data(symbol=symbol, timeframe=timeframe, days=days)
    if df is None or df.empty:
        logger.error(f"Failed to fetch historical data for {symbol} on {timeframe}")
        return {"success": False, "error": "Failed to fetch data"}

    print(f"✓ Fetched {len(df)} candles for {symbol} on {timeframe}")

    # Prepare data
    feature_columns = ['open', 'high', 'low', 'close', 'tick_volume']
    if 'spread' in df.columns:
        feature_columns.append('spread')

    print("🔄 Preparing data for LSTM...")
    X, y, scaler = prepare_data(df, seq_len, feature_columns=feature_columns)
    if len(X) == 0:
        logger.error(f"Insufficient data to train LSTM for {symbol} on {timeframe}")
        return {"success": False, "error": "Insufficient data"}

    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"✓ Data split: {len(X_train)} training samples, {len(X_test)} test samples")

    # Create TensorFlow dataset for efficient training
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Hyperparameter tuning with Keras Tuner
    print("\n🔍 Starting hyperparameter search...")
    total_trials = 3  # Reduced number of trials
    trial_times = []

    tuner = kt.Hyperband(lambda hp: build_lstm_model(hp, seq_len, X.shape[2]),
                         objective='val_loss', max_epochs=epochs,
                         directory=str(results_dir / 'tuner'),
                         project_name=f'lstm_{symbol}_{timeframe}')

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # Reduced patience
        ModelCheckpoint(
            filepath=str(model_dir / f"lstm_{symbol}_{timeframe}_checkpoint.h5"),
            save_best_only=True, monitor='val_loss'),
        TrialProgressCallback(trial_num=1, total_trials=total_trials)  # Add resource monitoring
    ]

    # Search for best hyperparameters with progress tracking
    print(f"\n📊 Running {total_trials} trials for hyperparameter optimization")
    print(f"Estimated completion time: {int(total_trials * 1)} - {int(total_trials * 3)} minutes\n")

    trial_start_time = time.time()

    # Wrapping tuner search with custom progress
    with tqdm(total=total_trials, desc="Training Progress", unit="trial",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

        original_on_trial_end = tuner.on_trial_end

        def on_trial_end_wrapper(trial):
            nonlocal trial_start_time
            trial_end_time = time.time()
            trial_duration = trial_end_time - trial_start_time
            trial_times.append(trial_duration)

            # Calculate average time and ETA
            avg_trial_time = sum(trial_times) / len(trial_times)
            completed_trials = len(trial_times)
            remaining_trials = total_trials - completed_trials
            eta_seconds = remaining_trials * avg_trial_time
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)

            pbar.set_postfix_str(f"Trial time: {trial_duration:.1f}s | ETA: {eta_time.strftime('%H:%M:%S')}")
            pbar.update(1)

            trial_start_time = time.time()
            original_on_trial_end(trial)

        tuner.on_trial_end = on_trial_end_wrapper

        tuner.search(train_dataset, validation_data=test_dataset, callbacks=callbacks, verbose=verbose)

    # Get the best model
    print("\n🏆 Retrieving best model...")
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Evaluate model
    metrics = evaluate_model(best_model, X_test, y_test, scaler, feature_columns,
                             str(results_dir), symbol)

    print("\n📊 Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    # Save final model
    model_path = model_dir / f"lstm_{symbol}_{timeframe}.h5"
    best_model.save(str(model_path))
    print(f"\n💾 LSTM model saved to {model_path}")

    # Save results summary
    with open(str(results_dir / f"results_{symbol}_{timeframe}.txt"), 'w') as f:
        f.write(f"LSTM Training Results for {symbol} on {timeframe}\n")
        f.write(f"Sequence Length: {seq_len}\n")
        f.write(f"Best Hyperparameters: {best_hyperparameters.values}\n")
        f.write(f"Data Points: {len(df)}\n")
        f.write(f"Features: {feature_columns}\n\n")
        f.write("Evaluation Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")

    total_time = time.time() - training_start_time

    print(f"\n{'=' * 60}")
    print(f"✅ TRAINING COMPLETED")
    print(f"Total Time: {total_time / 60:.1f} minutes")
    print(f"Model Path: {model_path}")
    print(f"{'=' * 60}\n")

    return {"success": True, "model_path": str(model_path), "metrics": metrics,
            "data_points": len(df), "sequence_length": seq_len}

def main() -> None:
    """Main function to train LSTM model from command line arguments."""
    parser = argparse.ArgumentParser(description="Train LSTM model for trading strategy")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., XAUUSD)")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (e.g., H1)")
    parser.add_argument("--days", type=int, default=500, help="Number of days of historical data")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for LSTM")
    parser.add_argument("--output", type=str, default=".", help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0=silent, 1=progress, 2=detailed)")
    args = parser.parse_args()

    train_and_save_model(symbol=args.symbol, timeframe=args.timeframe, days=args.days,
                         seq_len=args.seq_len, output_dir=args.output,
                         epochs=args.epochs, batch_size=args.batch_size,
                         verbose=args.verbose)

if __name__ == "__main__":
    main()