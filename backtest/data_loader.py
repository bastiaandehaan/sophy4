# backtest/data_loader.py - English Version
from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Dict
import MetaTrader5 as mt5
import pandas as pd

try:
    from config import logger
except ImportError:
    try:
        from ..config import logger
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).parent.parent))
        from config import logger

TIMEFRAME_MAP: Dict[str, int] = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1}


def _convert_timeframe(timeframe: Union[str, int]) -> int:
    """Convert timeframe string to MT5 constant."""
    if isinstance(timeframe, int):
        return timeframe
    if isinstance(timeframe, str) and timeframe in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[timeframe]

    valid_timeframes = ", ".join(TIMEFRAME_MAP.keys())
    raise ValueError(
        f"Invalid timeframe: '{timeframe}'. Valid values: {valid_timeframes}")


def fetch_historical_data(symbol: str, timeframe: Union[str, int] = mt5.TIMEFRAME_D1,
                          days: int = 1095, end_date: Optional[datetime] = None) -> \
Optional[pd.DataFrame]:
    """
    Fetch historical price data from MetaTrader5.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'GER40.cash')
        timeframe: Chart timeframe
        days: Number of days of historical data
        end_date: End date for data range (default: now)

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    if not mt5.initialize():
        logger.error(f"MT5 initialization failed, error code: {mt5.last_error()}")
        return None

    # Validate symbol
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbol '{symbol}' not found in MT5. Check symbol name.")
        return None

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Cannot add symbol '{symbol}' to MarketWatch")
            return None

    # Convert and validate timeframe
    try:
        tf_value = _convert_timeframe(timeframe)
        tf_name = timeframe if isinstance(timeframe, str) else next(
            (name for name, value in TIMEFRAME_MAP.items() if value == timeframe),
            f"Timeframe_{timeframe}")
    except ValueError as e:
        logger.error(str(e))
        return None

    # Set date range
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    logger.info(
        f"Fetching data for {symbol}, timeframe: {tf_name}, from {start_date} to {end_date}")

    try:
        # Primary method: copy_rates_range
        rates = mt5.copy_rates_range(symbol, tf_value, start_date, end_date)

        if rates is None or len(rates) == 0:
            error_code = mt5.last_error()
            logger.warning(f"No data received for {symbol}, error code: {error_code}")
            logger.info(f"Trying alternative method with copy_rates_from_pos...")

            # Fallback method
            alt_rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, days)
            if alt_rates is None or len(alt_rates) == 0:
                logger.error(f"Alternative method also failed for {symbol}")
                return None

            logger.info(
                f"Alternative method successful: {len(alt_rates)} data points retrieved")
            rates = alt_rates

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.index.name = 'datetime'

        logger.info(f"Data range: from {df.index.min()} to {df.index.max()}")
        logger.info(
            f"Historical data loaded: {len(df)} rows, columns: {list(df.columns)}")

        return df

    except Exception as e:
        error_info = mt5.last_error()
        logger.error(
            f"Error fetching data for {symbol}: {str(e)}, MT5 error code: {error_info}")
        return None