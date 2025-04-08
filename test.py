import pandas as pd
import vectorbt as vbt
import logging
import inspect

# Logging configureren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Dummy data genereren
def load_dummy_data():
    dates = pd.date_range(start="2023-01-01", end="2025-04-08", freq="D")
    prices = pd.Series([100 + i * 0.1 for i in range(len(dates))], index=dates,
                       name="close")
    return prices


# Simpele Bollinger Bands strategie
def bollong_strategy(data, window=20, std_dev=1.5, sl_percent=0.01, tp_percent=0.02,
                     use_trailing_stop=False):
    sig = inspect.signature(bollong_strategy)
    logger.info(f"Functie signature: {sig}")
    logger.info(f"Aantal verwachte parameters: {len(sig.parameters)}")
    logger.info(
        f"Ontvangen argumenten: {data[:5]}, {window}, {std_dev}, {sl_percent}, {tp_percent}, {use_trailing_stop}")

    # Bollinger Bands berekenen
    bbands = vbt.BBANDS.run(data, window=window, nstd=std_dev)
    entries = data > bbands.upper  # LONG als prijs boven upper band
    exits = data < bbands.lower  # Exit als prijs onder lower band

    # Portfolio simuleren
    pf = vbt.Portfolio.from_signals(close=data, entries=entries, exits=exits,
        sl_stop=sl_percent, tp_stop=tp_percent,
        tsl_stop=0.01 if use_trailing_stop else None, freq="1D")
    return pf


# Testfunctie
def test_bollong_strategy():
    # 1. Controleer de data
    data = load_dummy_data()
    logger.info(f"Data geladen: {len(data)} rijen")
    logger.info(f"Data head: \n{data.head()}")
    logger.info(
        f"Data columns: {data.name if isinstance(data, pd.Series) else data.columns}")

    # 2. Vereenvoudigde strategie zonder trailing stop
    logger.info("Testen zonder trailing stop...")
    logger.info(
        "Aanroep met: data=data, window=20, std_dev=1.5, sl_percent=0.01, tp_percent=0.02, use_trailing_stop=False")
    pf_no_trailing = bollong_strategy(data=data, window=20, std_dev=1.5,
        sl_percent=0.01, tp_percent=0.02, use_trailing_stop=False)
    logger.info(f"Portfolio zonder trailing stop aangemaakt: {pf_no_trailing}")


if __name__ == "__main__":
    try:
        test_bollong_strategy()
    except Exception as e:
        logger.error(f"Algemene fout in test: {str(e)}")