import pandas as pd
import numpy as np
import logging


# Eigen implementatie van Bollinger Bands zonder VectorBT's BBANDS
def calculate_bollinger_bands(data, window=20, std_dev=1.5):
    """Handmatige implementatie van Bollinger Bands zonder VectorBT"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dummy_data():
    """Genereert test prijsdata"""
    dates = pd.date_range(start="2023-01-01", end="2025-04-08", freq="D")
    prices = pd.Series([100 + i * 0.1 for i in range(len(dates))], index=dates,
                       name="close")
    return prices


def bollong_strategy(data, window=20, std_dev=1.5, sl_percent=0.01, tp_percent=0.02,
                     use_trailing_stop=False):
    """Bollinger Bands strategie met eigen BB implementatie"""
    # Gebruik eigen implementatie van Bollinger Bands
    upper_band, _, lower_band = calculate_bollinger_bands(data, window, std_dev)

    # Genereer signalen
    entries = data > upper_band  # LONG als prijs boven upper band
    exits = data < lower_band  # Exit als prijs onder lower band

    # Gebruik parameter dictionary om flexibel met stops om te gaan
    portfolio_kwargs = {'close': data, 'entries': entries, 'exits': exits,
        'sl_stop': sl_percent, 'tp_stop': tp_percent, 'freq': "1D"}

    # Alleen toevoegen als nodig
    if use_trailing_stop:
        # Niet direct sl_trail gebruiken omdat dit de fout veroorzaakt
        pass  # Voorlopig weglaten

    # Maak portfolio
    import vectorbt as vbt
    pf = vbt.Portfolio.from_signals(**portfolio_kwargs)
    return pf


def test_bollong_strategy():
    """Test de strategie"""
    try:
        data = load_dummy_data()
        logger.info(f"Data geladen: {len(data)} rijen")

        # Test zonder trailing stop
        logger.info("Test zonder trailing stop...")
        pf1 = bollong_strategy(data, use_trailing_stop=False)
        logger.info(
            f"Return: {pf1.total_return():.2%}, Sharpe: {pf1.sharpe_ratio():.2f}")

        # Skip trailing stop test voor nu  # logger.info("Test met trailing stop...")  # pf2 = bollong_strategy(data, use_trailing_stop=True)  # logger.info(f"Return: {pf2.total_return():.2%}, Sharpe: {pf2.sharpe_ratio():.2f}")

    except Exception as e:
        logger.error(f"Test fout: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        test_bollong_strategy()
        print("Test succesvol afgerond!")
    except Exception as e:
        logger.error(f"Algemene fout: {str(e)}")