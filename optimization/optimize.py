import vectorbt as vbt
import pandas as pd
import numpy as np
from utils.data_utils import fetch_historical_data
from strategies.bollong import bollong_signals
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from config import SYMBOL, INITIAL_CAPITAL, FEES, logger, OUTPUT_DIR
from itertools import product  # Standard Python library for cartesian product
import time
import logging


def optimize_parameters():
    """
    Voer een grid search uit om de optimale parameters voor de bollong-strategie te vinden.
    Optimaliseert op Sharpe Ratio, met FTMO-compliance als voorwaarde.
    """
    start_time = time.time()

    # Haal data op
    df = fetch_historical_data(SYMBOL)
    if df is None:
        return

    # Definieer parameter ranges
    window_range = np.arange(20, 101, 10)  # 20, 30, ..., 100
    std_dev_range = np.arange(1.5, 3.1, 0.5)  # 1.5, 2.0, 2.5, 3.0
    sl_atr_mult_range = np.arange(1.0, 3.1, 0.5)  # 1.0, 1.5, ..., 3.0
    tp_atr_mult_range = np.arange(2.0, 5.1, 0.5)  # 2.0, 2.5, ..., 5.0

    # Maak parameter combinaties met Python's built-in product functie
    param_combinations = list(
        product(window_range, std_dev_range, sl_atr_mult_range, tp_atr_mult_range))

    total_combinations = len(param_combinations)
    logger.info(
        f"Optimalisatie gestart met {total_combinations} parameter combinaties...")

    # Tijdelijk verlaag log level om externe logs te onderdrukken
    original_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)  # Alleen errors tonen tijdens berekeningen

    best_sharpe = -np.inf
    best_params = None
    best_metrics = None
    compliant_count = 0
    tested_count = 0

    # Loop door alle combinaties
    try:
        for params in param_combinations:
            window, std_dev, sl_atr_mult, tp_atr_mult = params
            tested_count += 1

            # Toon voortgangsindicator voor elke 10% van verwerkte combinaties
            if tested_count % max(1,
                                  total_combinations // 10) == 0 or tested_count == 1:
                progress = tested_count / total_combinations * 100
                elapsed = time.time() - start_time
                remaining = (elapsed / tested_count) * (
                        total_combinations - tested_count) if tested_count > 0 else 0
                print(
                    f"Voortgang: {progress:.1f}% | Verwerkt: {tested_count}/{total_combinations} | "
                    f"Tijd: {elapsed:.1f}s | Resterende tijd: ~{remaining:.1f}s",
                    end='\r')

            # Genereer signalen
            entries, sl_stop, tp_stop = bollong_signals(df, int(window), std_dev,
                                                        sl_atr_mult, tp_atr_mult)

            # Voer backtest uit
            pf = vbt.Portfolio.from_signals(close=df['close'], entries=entries,
                                            sl_stop=sl_stop, tp_stop=tp_stop,
                                            init_cash=INITIAL_CAPITAL, fees=FEES,
                                            freq='1D')

            # Bereken metrics
            metrics = {'total_return': pf.total_return(),
                       'sharpe_ratio': pf.sharpe_ratio(),
                       'max_drawdown': pf.max_drawdown(),
                       'win_rate': pf.trades.win_rate() if len(pf.trades) > 0 else 0,
                       'trades_count': len(pf.trades)}

            # Check FTMO compliance
            compliant, profit_reached = check_ftmo_compliance(pf, metrics)

            # Optimaliseer op Sharpe Ratio, maar alleen als FTMO-compliant
            if compliant and profit_reached:
                compliant_count += 1
                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    best_params = params
                    best_metrics = metrics
    finally:
        # Zet log level terug naar origineel
        logger.setLevel(original_level)

    # Print een lege regel om voortgangsindicator te wissen
    print()

    # Log samenvattend resultaat
    logger.info("\n" + "=" * 60)
    logger.info(f"OPTIMALISATIE SAMENVATTING")
    logger.info("=" * 60)
    logger.info(f"Totaal geteste combinaties: {tested_count}/{total_combinations}")
    logger.info(
        f"FTMO-compliant combinaties: {compliant_count} ({compliant_count / tested_count * 100:.1f}%)")
    logger.info(f"Totale tijd: {time.time() - start_time:.1f} seconden")

    # Log beste resultaat
    if best_params is not None:
        window, std_dev, sl_atr_mult, tp_atr_mult = best_params
        logger.info("\nBESTE PARAMETERS:")
        logger.info(f"  Window:      {window}")
        logger.info(f"  Std Dev:     {std_dev}")
        logger.info(f"  SL ATR Mult: {sl_atr_mult}")
        logger.info(f"  TP ATR Mult: {tp_atr_mult}")

        logger.info("\nBESTE RESULTATEN:")
        logger.info(f"  Return:      {best_metrics['total_return']:.2%}")
        logger.info(f"  Sharpe:      {best_metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Drawdown:    {best_metrics['max_drawdown']:.2%}")
        logger.info(f"  Win Rate:    {best_metrics['win_rate']:.2%}")
        logger.info(f"  Trades:      {best_metrics['trades_count']}")
    else:
        logger.info("\nGeen FTMO-compliant parameters gevonden.")

    logger.info("=" * 60)


if __name__ == "__main__":
    optimize_parameters()