from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any

import MetaTrader5 as mt5
import pandas as pd

from config import logger

# Timeframe mapping van string naar MT5 constante
TIMEFRAME_MAP: Dict[str, int] = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1}


def _convert_timeframe(timeframe: Union[str, int]) -> int:
    """
    Converteer een timeframe string naar de bijbehorende MT5 constante.

    Args:
        timeframe: String (bijv. 'D1') of MT5 timeframe constante.

    Returns:
        MT5 timeframe constante (int).

    Raises:
        ValueError: Als de gegeven timeframe string ongeldig is.
    """
    # Als het al een integer is, neem aan dat het een geldige MT5 constante is
    if isinstance(timeframe, int):
        return timeframe

    # Als het een string is, zoek in de mapping
    if isinstance(timeframe, str) and timeframe in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[timeframe]

    # Ongeldig timeframe
    valid_timeframes = ", ".join(TIMEFRAME_MAP.keys())
    raise ValueError(
        f"Ongeldige timeframe: '{timeframe}'. Geldige waarden zijn: {valid_timeframes}")


def fetch_historical_data(symbol: str, timeframe: Union[str, int] = mt5.TIMEFRAME_D1,
        days: int = 1095, end_date: Optional[datetime] = None) -> Optional[
    pd.DataFrame]:
    """
    Haalt historische data op via MetaTrader 5.

    Args:
        symbol: Trading symbool (bijv. 'GER40.cash').
        timeframe: MT5 timeframe constante of string (bijv. mt5.TIMEFRAME_D1 of 'D1').
        days: Aantal dagen historische data.
        end_date: Einddatum (standaard: nu).

    Returns:
        DataFrame met OHLC-data of None bij fout.
    """
    # MT5 initialisatie controleren
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt, error code: {mt5.last_error()}")
        return None

    # Controleer of symbool beschikbaar is
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbool '{symbol}' niet gevonden in MT5. Controleer de naam.")
        return None

    # Zorg ervoor dat het symbool zichtbaar is in MarketWatch
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Kan symbool '{symbol}' niet toevoegen aan MarketWatch")
            return None

    # Converteer timeframe indien nodig
    try:
        tf_value = _convert_timeframe(timeframe)
        tf_name = timeframe if isinstance(timeframe, str) else next(
            (name for name, value in TIMEFRAME_MAP.items() if value == timeframe),
            f"Timeframe_{timeframe}")
    except ValueError as e:
        logger.error(str(e))
        return None

    # Datums instellen
    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(
        f"Data ophalen voor {symbol}, timeframe: {tf_name}, van {start_date} tot {end_date}")

    try:
        # Probeer data op te halen met expliciete tijdzone-agnostische datums
        rates = mt5.copy_rates_range(symbol, tf_value, start_date, end_date)

        if rates is None or len(rates) == 0:
            error_code = mt5.last_error()
            logger.error(f"Geen data ontvangen voor {symbol}, error code: {error_code}")

            # Probeer alternatieve methode als rates_range mislukt
            logger.info(f"Probeer alternatieve methode met copy_rates_from...")
            # Haal maximaal aantal beschikbare datapunten op
            alt_rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, days)
            if alt_rates is None or len(alt_rates) == 0:
                logger.error(f"Alternatieve methode ook mislukt voor {symbol}")
                return None

            logger.info(
                f"Alternatieve methode succesvol: {len(alt_rates)} datapunten opgehaald")
            rates = alt_rates

        # Converteer naar DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.index.name = 'datetime'

        # Log beschikbare kolommen voor debugging
        logger.info(
            f"Historische data geladen: {len(df)} rijen, kolommen: {list(df.columns)}")
        return df

    except Exception as e:
        error_info = mt5.last_error()
        logger.error(
            f"Fout bij ophalen data voor {symbol}: {str(e)}, MT5 error code: {error_info}")
        return None


def fetch_live_data(symbol: str, timeframe: Union[str, int] = mt5.TIMEFRAME_D1) -> \
Optional[pd.DataFrame]:
    """
    Haalt de meest recente candle op via MetaTrader 5.

    Args:
        symbol: Trading symbool (bijv. 'GER40.cash').
        timeframe: MT5 timeframe constante of string (bijv. mt5.TIMEFRAME_D1 of 'D1').

    Returns:
        DataFrame met de laatste candle of None bij fout.
    """
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt, error code: {mt5.last_error()}")
        return None

    # Converteer timeframe indien nodig
    try:
        tf_value = _convert_timeframe(timeframe)
    except ValueError as e:
        logger.error(str(e))
        return None

    try:
        rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 1)  # Laatste candle
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