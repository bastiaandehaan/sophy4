from datetime import datetime, timedelta
from typing import Optional, Union

import MetaTrader5 as mt5
import pandas as pd

from config import logger


def fetch_historical_data(symbol: str, timeframe: int = mt5.TIMEFRAME_D1,
                          days: int = 1095, end_date: Optional[datetime] = None) -> \
Optional[pd.DataFrame]:
    """
    Haalt historische data op via MetaTrader 5.

    Args:
        symbol: Trading symbool (bijv. 'GER40.cash').
        timeframe: MT5 timeframe constante (bijv. mt5.TIMEFRAME_D1).
        days: Aantal dagen historische data.
        end_date: Einddatum (standaard: nu).

    Returns:
        DataFrame met OHLC-data of None bij fout.
    """
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt, error code: {mt5.last_error()}")
        return None

    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(
        f"Data ophalen voor {symbol}, timeframe: {timeframe}, van {start_date} tot {end_date}")

    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            logger.error(
                f"Geen data ontvangen voor {symbol}, error code: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.index.name = 'datetime'

        # Log beschikbare kolommen voor debugging
        logger.info(
            f"Historische data geladen: {len(df)} rijen, kolommen: {list(df.columns)}")
        return df

    except Exception as e:
        logger.error(
            f"Fout bij ophalen data voor {symbol}: {str(e)}, MT5 error code: {mt5.last_error()}")
        return None


def fetch_live_data(symbol: str, timeframe: int = mt5.TIMEFRAME_D1) -> Optional[
    pd.DataFrame]:
    """
    Haalt de meest recente candle op via MetaTrader 5.

    Args:
        symbol: Trading symbool (bijv. 'GER40.cash').
        timeframe: MT5 timeframe constante (bijv. mt5.TIMEFRAME_D1).

    Returns:
        DataFrame met de laatste candle of None bij fout.
    """
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt, error code: {mt5.last_error()}")
        return None

    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)  # Laatste candle
        if rates is None or len(rates) == 0:
            logger.error(
                f"Geen live data voor {symbol}, error code: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.index.name = 'datetime'

        logger.info(
            f"Live data geladen voor {symbol}: {len(df)} rijen, kolommen: {list(df.columns)}")
        return df

    except Exception as e:
        logger.error(
            f"Fout bij ophalen live data voor {symbol}: {str(e)}, MT5 error code: {mt5.last_error()}")
        return None
