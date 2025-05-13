import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# MT5 initialization
if not mt5.initialize():
    raise ConnectionError("MT5 initialization failed")


# Fetch historical data from MT5
def fetch_data(symbol, start_date, end_date, timeframe=mt5.TIMEFRAME_H1):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt.timestamp(),
                                 end_dt.timestamp())
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data retrieved for {symbol}")
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    data.rename(
        columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'},
        inplace=True)
    return data


# Calculate technical indicators
def calculate_indicators(data):
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = 100 - (
                100 / (1 + data['Close'].pct_change().rolling(window=14).mean()))
    data['Bollinger_Bands'] = data['Close'].rolling(window=20).mean() + 2 * data[
        'Close'].rolling(window=20).std()
    data['ATR'] = data['High'].rolling(window=14).max() - data['Low'].rolling(
        window=14).min()
    return data


# Perform sentiment analysis
def sentiment_analysis(data):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(str(idx))['compound'] for idx in data.index]
    data['Sentiment'] = sentiment_scores
    return data


# Prepare data for training
def prepare_data(data):
    data.dropna(inplace=True)
    X = data[['MA_50', 'MA_200', 'RSI', 'Bollinger_Bands', 'ATR', 'Sentiment']]
    y = data['Close']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                        random_state=42)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    return X_train, X_test, y_train, y_test, scaler


# Build and train the model
def build_model(X_train, y_train):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, return_sequences=True),
                            input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=50)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss',
                                       save_best_only=True, mode='min')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
              callbacks=[early_stopping, model_checkpoint])
    return model


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return np.sqrt(mse)


# Main execution
if __name__ == "__main__":
    symbol = 'GER40.cash'
    start_date = '2019-02-26'  # Adjusted to fit 1095 days until 2022-02-26
    end_date = '2022-02-26'

    data = fetch_data(symbol, start_date, end_date)
    data = calculate_indicators(data)
    data = sentiment_analysis(data)
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)
    model = build_model(X_train, y_train)
    rmse = evaluate_model(model, X_test, y_test)
    print(f'RMSE: {rmse:.2f}')

    # Save the model for later use
    model.save('dax_lstm_model.h5')
    mt5.shutdown()