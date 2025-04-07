# Sophy4/strategies/bollong.py
import pandas as pd
import numpy as np
import vectorbt as vbt
from config import logger


def calculate_atr(df, window=14):
    """
    Bereken de Average True Range (ATR) voor dynamische stop-loss en take-profit.
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr = pd.DataFrame(
        {'tr1': high - low, 'tr2': abs(high - close), 'tr3': abs(low - close)}).max(
        axis=1)
    return tr.rolling(window=window).mean()


def bollong_signals(df, window=50, std_dev=2.0, sl_atr_mult=2.0, tp_atr_mult=3.0):
    """
    Genereer long-only Bollinger Bands breakout signalen met ATR-gebaseerde stops.

    Args:
        df: DataFrame met OHLC-data
        window: Bollinger Bands window
        std_dev: Aantal standaarddeviaties voor de bands
        sl_atr_mult: ATR-multiplier voor stop-loss
        tp_atr_mult: ATR-multiplier voor take-profit

    Returns:
        entries: Series met entry-signalen (True/False)
        sl_stop: Series met stop-loss percentages
        tp_stop: Series met take-profit percentages
    """
    # Bereken Bollinger Bands
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)

    # Bereken ATR
    atr = calculate_atr(df)

    # Genereer long-only entry-signalen
    entries = df['close'] > upper_band

    # Bereken stop-loss en take-profit als percentages
    sl_stop = sl_atr_mult * atr / df['close']
    tp_stop = tp_atr_mult * atr / df['close']

    # Log aantal signalen
    logger.info(f"Aantal LONG signalen: {entries.sum()}")

    return entries, sl_stop, tp_stop