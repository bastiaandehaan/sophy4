# optimization/optimize.py
import argparse
import json
import sys
import time
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqdm import tqdm

# Minimaliseer logging nog meer
logging.getLogger('vectorbt').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import fetch_historical_data
from strategies import get_strategy, STRATEGIES
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from config import SYMBOL, INITIAL_CAPITAL, FEES, OUTPUT_DIR
import MetaTrader5 as mt5


def optimize_strategy(strategy_name, symbol=SYMBOL, symbols=None, timeframe=None,
        metric="sharpe_ratio", top_n=3, initial_capital=INITIAL_CAPITAL, fees=FEES,
        period_days=1095, ftmo_compliant_only=False, full_analysis=False,
        monte_carlo=False):
    # Als symbols niet expliciet gegeven, gebruik de standaard symbol
    if symbols is None:
        symbols = [symbol]

    # Initialiseer timeframe mapping
    tf_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1}

    # Converteer timeframe indien string
    if isinstance(timeframe, str):
        timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_D1)

    # Resultaten voor alle symbolen
    all_symbol_results = {}

    for current_symbol in symbols:
        start_time = time.time()
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True, parents=True)

        # Strategie ophalen
        if strategy_name not in STRATEGIES:
            print(f"Strategie '{strategy_name}' niet gevonden")
            continue

        strategy_class = STRATEGIES[strategy_name]
        param_ranges = strategy_class.get_default_params()

        # Data ophalen
        df = fetch_historical_data(current_symbol, timeframe=timeframe,
                                   days=period_days)
        if df is None:
            print(f"Geen data voor {current_symbol}")
            continue

        # Parameter combinaties genereren
        param_names = list(param_ranges.keys())
        param_values = [
            param_ranges[name] if isinstance(param_ranges[name], list) else [
                param_ranges[name]] for name in param_names]
        param_combinations = list(product(*param_values))
        total_combinations = len(param_combinations)

        results = []
        compliant_count = 0

        # Gebruik tqdm voor overall voortgangsbalk met minimale output
        with tqdm(total=total_combinations, desc=f"Optimalisatie {current_symbol}",
                  unit="combinatie",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))

                try:
                    strategy = get_strategy(strategy_name, **param_dict)
                    strategy.validate_parameters()

                    # Genereer signalen
                    entries, sl_stop, tp_stop = strategy.generate_signals(df.copy())

                    # Portfolio aanmaken
                    portfolio_kwargs = {'close': df['close'], 'entries': entries,
                        'sl_stop': sl_stop, 'tp_stop': tp_stop,
                        'init_cash': initial_capital, 'fees': fees, 'freq': '1D'}

                    pf = vbt.Portfolio.from_signals(**portfolio_kwargs)

                    # Bereken metrics
                    metrics_dict = {'total_return': float(pf.total_return()),
                        'sharpe_ratio': float(pf.sharpe_ratio()),
                        'max_drawdown': float(pf.max_drawdown()), }

                    # Controleer FTMO-compliance als gevraagd
                    if ftmo_compliant_only:
                        compliant, profit_target = check_ftmo_compliance(pf,
                                                                         metrics_dict)
                        if not compliant:
                            pbar.update(1)
                            continue
                        compliant_count += 1

                    results.append({'params': param_dict, 'metrics': metrics_dict})

                except Exception:
                    pass  # Negeer fouten zonder output

                pbar.update(1)

        # Sorteer resultaten op de gespecificeerde metric
        sorted_results = sorted(results, key=lambda x: x['metrics'][metric],
                                reverse=metric not in ['max_drawdown'])
        top_results = sorted_results[:top_n]

        # Sla resultaten op
        timeframe_str = f"_{timeframe}" if timeframe else ""
        results_file = output_path / f"{strategy_name}_{current_symbol}{timeframe_str}_optimized.json"
        with open(results_file, 'w') as f:
            json.dump(top_results, f, indent=2)

        all_symbol_results[current_symbol] = top_results
        print(
            f"Optimalisatie voltooid voor {current_symbol} in {time.time() - start_time:.1f}s")

    return all_symbol_results


def main():
    parser = argparse.ArgumentParser(description="Sophy4 Strategie Optimizer")

    # Basis argumenten
    parser.add_argument("--strategy", type=str, default="BollongStrategy",
                        help="Naam van de strategie")
    parser.add_argument("--symbol", type=str, default=SYMBOL,
                        help=f"Primair trading symbool (default: {SYMBOL})")
    parser.add_argument("--symbols", nargs='+', help="Meerdere symbolen om te testen")
    parser.add_argument("--timeframe", type=str, default='D1',
                        help="Timeframe voor analyse (default: D1)")

    # Optimalisatie parameters
    parser.add_argument("--metric", type=str, default="sharpe_ratio",
                        help="Metric om te optimaliseren")
    parser.add_argument("--top_n", type=int, default=3, help="Aantal top resultaten")
    parser.add_argument("--initial_capital", type=float, default=INITIAL_CAPITAL,
                        help="Initieel kapitaal voor backtest")
    parser.add_argument("--fees", type=float, default=FEES, help="Transactiekosten")
    parser.add_argument("--period_days", type=int, default=1095,
                        help="Aantal dagen historische data")

    # Geavanceerde opties
    parser.add_argument("--ftmo_only", action="store_true",
                        help="Alleen FTMO-compliant resultaten")
    parser.add_argument("--full_analysis", action="store_true",
                        help="Voer volledige analyse uit")
    parser.add_argument("--monte_carlo", action="store_true",
                        help="Voer Monte Carlo analyse uit")

    args = parser.parse_args()

    # Bereid symbolen voor
    symbols = args.symbols or [args.symbol]

    # Voer optimalisatie uit
    results = optimize_strategy(strategy_name=args.strategy, symbols=symbols,
        timeframe=args.timeframe, metric=args.metric, top_n=args.top_n,
        initial_capital=args.initial_capital, fees=args.fees,
        period_days=args.period_days, ftmo_compliant_only=args.ftmo_only,
        full_analysis=args.full_analysis, monte_carlo=args.monte_carlo)

    # Print de top resultaten
    for symbol, symbol_results in results.items():
        print(f"\nTop resultaten voor {symbol}:")
        for i, result in enumerate(symbol_results, 1):
            print(f"\n#{i} Parameter Set:")
            print("  Parameters:", result['params'])
            print("  Metrics:")
            for metric, value in result['metrics'].items():
                print(f"    {metric}: {value:.4f}")

    print("\nOptimalisatie voltooid!")


if __name__ == "__main__":
    main()