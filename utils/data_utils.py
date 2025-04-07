# Sophy4/utils/data_utils.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from config import logger

def fetch_historical_data(symbol, timeframe=mt5.TIMEFRAME_D1, days=1095, end_date=None):
    if not mt5.initialize():
        logger.error("MT5 initialisatie mislukt")
        return None
    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        logger.error(f"Geen data voor {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    logger.info(f"Historische data geladen: {len(df)} rijen")
    return df

def fetch_live_data(symbol, timeframe=mt5.TIMEFRAME_D1):
    if not mt5.initialize():
        logger.error("MT5 initialisatie mislukt")
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)  # Laatste candle
    if rates is None or len(rates) == 0:
        logger.error(f"Geen live data voor {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df