import logging
import typing as t
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 1.5,
        min_periods: t.Optional[int] = None) -> t.Tuple[
    pd.Series, pd.Series, pd.Series]:
    """
    Bereken Bollinger Bands met verbeterde foutafhandeling en type hints.

    Args:
        data (pd.Series): Input prijsserie
        window (int, optional): Rolvenster voor berekening. Defaults to 20.
        std_dev (float, optional): Aantal standaarddeviaties. Defaults to 1.5.
        min_periods (int, optional): Minimum aantal perioden voor berekening. Defaults to window.

    Returns:
        Tuple van (upper band, midden band (SMA), lower band)

    Raises:
        TypeError: Als input geen pandas Series is
        ValueError: Als window of std_dev ongeldig is
    """
    # Input validatie
    if not isinstance(data, pd.Series):
        raise TypeError(f"Input moet een pandas Series zijn, niet {type(data)}")

    if window <= 0:
        raise ValueError(f"Window moet positief zijn, niet {window}")

    if std_dev <= 0:
        raise ValueError(f"Standaarddeviatie moet positief zijn, niet {std_dev}")

    # Gebruik min_periods om berekening aan te passen
    min_periods = min_periods or window

    try:
        # Bereken Bollinger Bands met verbeterde rolling window
        rolling_mean = data.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = data.rolling(window=window, min_periods=min_periods).std()

        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)

        # Log details van de berekening
        logger.info(f"Bollinger Bands berekend: "
                    f"window={window}, std_dev={std_dev}, min_periods={min_periods}")

        return upper_band, rolling_mean, lower_band

    except Exception as e:
        logger.error(f"Fout bij berekenen Bollinger Bands: {e}")
        raise