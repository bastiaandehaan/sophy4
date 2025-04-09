#!/usr/bin/env python
# optimization/optimize.py
"""
Een verbeterde optimalisatie tool voor Sophy4 met beknopte output.
"""
import argparse
import itertools
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Voeg projectroot toe aan Python path
sys.path.append(str(Path(__file__).parent.parent))

from strategies import get_strategy, STRATEGIES
from utils.data_utils import fetch_historical_data
from utils.backtest import run_backtest
from config import logger, OUTPUT_DIR

# Schakel alle loggers uit behalve onze eigen
for name in logging.root.manager.loggerDict:
    if name != "summary":
        logging.getLogger(name).setLevel(logging.ERROR)

# Maak een aparte console handler voor samenvattingen
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')  # Alleen het bericht, geen datum en niveau
console_handler.setFormatter(formatter)

# Maak een summary logger
summary_logger = logging.getLogger("summary")
summary_logger.setLevel(logging.INFO)
for handler in summary_logger.handlers:
    summary_logger.removeHandler(handler)
summary_logger.addHandler(console_handler)
summary_logger.propagate = False  # Voorkom dubbele logs


def update_progress(current, total, bar_length=50):
    """Toon een voortgangsbalk in de console."""
    percent = float(current) / total
    arrow = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write(f"\r|{arrow}{spaces}| {current}/{total} ({percent:.1%})")
    sys.stdout.flush()


def quick_optimize(strategy_name, symbol, timeframe="D1", days=365,
                   metric="sharpe_ratio", top_n=5, verbose=False, quick=False):
    """
    Voert een snelle parameter optimalisatie uit en slaat de resultaten op.
    Toont alleen een beknopte samenvatting van de resultaten.

    Args:
        strategy_name: Naam van de strategie
        symbol: Handelssymbool
        timeframe: Timeframe ("D1", "H4", etc)
        days: Aantal dagen historische data
        metric: Metric om op te optimaliseren
        top_n: Aantal beste resultaten om te tonen
        verbose: Toon uitgebreide logs
        quick: Gebruik een beperkte parameter grid (sneller)
    """
    start_time = time.time()

    if verbose:
        # Herstel normale logging als verbose modus actief is
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)

    summary_logger.info(f"\n{'=' * 80}")
    summary_logger.info(
        f"OPTIMALISATIE GESTART: {strategy_name} op {symbol} ({timeframe})")
    summary_logger.info(f"{'=' * 80}")

    # Controleer of de strategie bestaat
    if strategy_name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        summary_logger.error(
            f"Strategie {strategy_name} niet gevonden. Beschikbaar: {available}")
        return []

    # Haal de standaard parameters op
    strategy_class = STRATEGIES[strategy_name]
    param_grid = strategy_class.get_default_params()

    # Beperk parameters voor snelle run
    if quick:
        for key, values in param_grid.items():
            if isinstance(values, list) and len(values) > 3:
                # Neem alleen begin, midden en eind waarde
                param_grid[key] = [values[0], values[len(values) // 2], values[-1]]

    # Bepaal alle parameter combinaties
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    total_combinations = len(param_combinations)

    summary_logger.info(f"Testen van {total_combinations} parameter combinaties...")

    # Haal historische data op
    df = fetch_historical_data(symbol, timeframe=timeframe, days=days)
    if df is None or df.empty:
        summary_logger.error(f"Geen data beschikbaar voor {symbol}")
        return []

    # Test elke parameter combinatie
    results = []
    result_hash = set()  # Voorkom duplicaten

    try:
        summary_logger.info(f"0% |{' ' * 50}| 100%")

        for i, values in enumerate(param_combinations):
            params = dict(zip(param_names, values))

            # Toon voortgang
            update_progress(i + 1, total_combinations)

            try:
                # Maak strategie instantie
                strategy = get_strategy(strategy_name, **params)

                # Genereer signalen
                entries, sl_stop, tp_stop = strategy.generate_signals(df)
                if entries.sum() == 0:
                    continue

                # Maak een kopie van het dataframe
                backtest_df = df.copy()
                backtest_df['entries'] = entries
                backtest_df['sl_stop'] = sl_stop
                backtest_df['tp_stop'] = tp_stop

                # Voer backtest uit
                pf, metrics = run_backtest(backtest_df, symbol, strategy_params=params)

                # Als metric niet bestaat in de resultaten, gebruik een fallback
                if metric not in metrics and len(metrics) > 0:
                    if verbose:
                        summary_logger.warning(
                            f"Metric {metric} niet gevonden, gebruik {list(metrics.keys())[0]}")
                    metric = list(metrics.keys())[0]

                # Voeg aantal signalen toe aan metrics
                metrics['signal_count'] = int(entries.sum())
                metrics['trade_count'] = metrics.get('trades_count', 0)

                # Controleer op duplicaten door een hash van de belangrijkste metrics te maken
                # Dit is om duplicaten te vermijden in de resultaten
                result_key = (round(metrics.get('sharpe_ratio', 0), 3),
                              round(metrics.get('total_return', 0), 3),
                              round(metrics.get('max_drawdown', 0), 3),
                              metrics.get('trade_count', 0))

                if result_key not in result_hash:
                    result_hash.add(result_key)
                    # Sla resultaten op
                    results.append({'params': params, 'metrics': metrics})

            except Exception as e:
                if verbose:
                    logger.error(f"Fout bij testen parameters {params}: {str(e)}")

        # Nieuwe regel na progress bar
        print()

    except KeyboardInterrupt:
        print("\nOptimalisatie onderbroken door gebruiker!")

    # Sorteer op de gekozen metric
    if results:
        sorted_results = sorted(results,
            key=lambda x: x['metrics'].get(metric, -999999), reverse=True)
        top_results = sorted_results[:top_n]

        # Log de top resultaten in een nette tabel
        summary_logger.info(f"\n{'=' * 80}")
        summary_logger.info(
            f"TOP {len(top_results)} RESULTATEN VOOR {strategy_name} OP {symbol}:")
        summary_logger.info(f"{'=' * 80}")

        # Maak een tabelkop
        summary_logger.info(
            f"{'#':<3} {'Sharpe':<8} {'Return':<8} {'DrawDn':<8} {'WinRate':<8} {'Trades':<6} {'Params'}")
        summary_logger.info(f"{'-' * 80}")

        # Log elke top combinatie
        for i, result in enumerate(top_results):
            metrics = result['metrics']
            params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items() if
                                    k in ['window', 'std_dev', 'sl_fixed_percent',
                                          'tp_fixed_percent', 'use_trailing_stop',
                                          'trailing_stop_percent']])

            summary_logger.info(f"{i + 1:<3} "
                                f"{metrics.get('sharpe_ratio', 0):.2f}     "
                                f"{metrics.get('total_return', 0) * 100:.2f}%    "
                                f"{metrics.get('max_drawdown', 0) * 100:.2f}%    "
                                f"{metrics.get('win_rate', 0) * 100:.2f}%    "
                                f"{metrics.get('trade_count', 0):<6} "
                                f"{params_str}")

        # Sla resultaten op
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True, parents=True)

        output_file = output_path / f"{strategy_name}_{symbol}_quick_optim.json"
        with open(output_file, 'w') as f:
            # Maak JSON-serializable
            serializable_results = []
            for result in top_results:
                metrics_clean = {k: float(v) if isinstance(v, (float, int)) else str(v)
                                 for k, v in result['metrics'].items()}
                serializable_results.append(
                    {'params': result['params'], 'metrics': metrics_clean})
            json.dump(serializable_results, f, indent=2)

        elapsed_time = time.time() - start_time
        summary_logger.info(f"\nResultaten opgeslagen in {output_file}")
        summary_logger.info(f"Optimalisatie voltooid in {elapsed_time:.1f} seconden.")
        return top_results

    summary_logger.warning("\nGeen resultaten gevonden")
    return []


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy4 Quick Optimization Tool")
    parser.add_argument("--strategy", type=str, required=True,
                        help="Strategie om te optimaliseren")
    parser.add_argument("--symbol", type=str, default="GER40.cash",
                        help="Handelssymbool")
    parser.add_argument("--timeframe", type=str, default="D1", help="Timeframe")
    parser.add_argument("--days", type=int, default=365,
                        help="Aantal dagen historische data")
    parser.add_argument("--metric", type=str, default="sharpe_ratio",
                        help="Metric om te optimaliseren")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Aantal beste parameter sets")
    parser.add_argument("--verbose", action="store_true",
                        help="Toon alle log berichten")
    parser.add_argument("--quick", action="store_true",
                        help="Gebruik een beperkte parameter grid (sneller)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quick_optimize(args.strategy, args.symbol, args.timeframe, args.days, args.metric,
                   args.top_n, args.verbose, args.quick)