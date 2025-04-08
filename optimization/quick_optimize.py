# optimization/quick_optimize.py
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt

# Voeg projectroot toe aan pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import fetch_historical_data
from strategies import get_strategy
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from config import SYMBOL, INITIAL_CAPITAL, FEES, OUTPUT_DIR


def quick_optimize(strategy_name, symbol=SYMBOL, top_n=3, period_days=1095):
    """
    Versimpelde optimalisatie functie die een beperkte set parameters test.
    """
    print(f"--- QUICK OPTIMIZE voor {strategy_name} op {symbol} ---")

    # Beperk parameter ranges drastisch voor snelle tests
    param_ranges = {
        'window': [20, 50, 80],
        'std_dev': [1.5, 2.5],
        'sl_method': ['fixed_percent'],
        'sl_fixed_percent': [0.02],
        'tp_method': ['fixed_percent'],
        'tp_fixed_percent': [0.04]
    }

    # Haal data op als een volledig nieuwe dataframe
    print("1. Data ophalen...")
    df_original = fetch_historical_data(symbol, days=period_days)
    if df_original is None:
        print("❌ Geen data gevonden.")
        return

    print(f"✓ Data geladen: {len(df_original)} rijen")

    # Maak parameter combinaties
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] if isinstance(param_ranges[name], list) else [param_ranges[name]] for name in param_names]
    param_combinations = list(product(*param_values))
    total_combinations = len(param_combinations)

    print(f"2. Start test met {total_combinations} parameter combinaties")

    # Resultaten opslaan
    results = []

    # Voer tests uit
    for i, params in enumerate(param_combinations):
        # Maak parameter dictionary
        param_dict = {name: value for name, value in zip(param_names, params)}

        print(f"\nTest {i+1}/{total_combinations}: {param_dict}")

        try:
            # Maak NIEUWE kopie van data voor elke test
            df = df_original.copy(deep=True)

            # Maak strategie instantie
            strategy = get_strategy(strategy_name, **param_dict)

            # Genereer signalen
            print("  Signalen genereren...")
            entries, sl_stop, tp_stop = strategy.generate_signals(df)

            # Zorg dat alle Series onafhankelijk zijn
            entries_copy = pd.Series(entries.values, index=df.index)
            sl_stop_copy = pd.Series(sl_stop.values, index=df.index)
            tp_stop_copy = pd.Series(tp_stop.values, index=df.index)

            # Run backtest
            print(f"  Backtest uitvoeren...")
            pf = vbt.Portfolio.from_signals(
                close=df['close'].values,  # Gebruik numpy array ipv pandas
                entries=entries_copy.values,  # Gebruik numpy arrays
                sl_stop=sl_stop_copy.values,
                tp_stop=tp_stop_copy.values,
                init_cash=INITIAL_CAPITAL,
                fees=FEES,
                freq='1D'
            )

            # Resultaten berekenen
            return_pct = pf.total_return()
            sharpe = pf.sharpe_ratio()
            drawdown = pf.max_drawdown()
            win_rate = pf.trades.win_rate() if len(pf.trades) > 0 else 0

            # FTMO compliance check
            compliant, profit_target = check_ftmo_compliance(pf, {
                'total_return': return_pct,
                'max_drawdown': drawdown
            })

            # Voeg resultaten toe
            results.append({
                'params': param_dict,
                'metrics': {
                    'total_return': float(return_pct),
                    'sharpe_ratio': float(sharpe),
                    'max_drawdown': float(drawdown),
                    'win_rate': float(win_rate),
                    'trades_count': len(pf.trades)
                },
                'ftmo_compliant': compliant,
                'profit_target_reached': profit_target
            })

            print(f"  ✓ Return: {return_pct:.2%}, Sharpe: {sharpe:.2f}, Drawdown: {drawdown:.2%}")
            print(f"  ✓ FTMO Compliant: {'JA' if compliant else 'NEE'}")

        except Exception as e:
            print(f"  ❌ Fout: {str(e)}")

    # Sorteer op sharpe ratio
    if results:
        sorted_results = sorted(results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
        top_results = sorted_results[:top_n]

        # Toon top resultaten
        print("\n--- TOP RESULTATEN ---")
        for i, result in enumerate(top_results):
            print(f"\n#{i+1}: Sharpe = {result['metrics']['sharpe_ratio']:.4f}")
            for param_name, param_value in result['params'].items():
                print(f"  {param_name}: {param_value}")
            print("  --- Performance ---")
            print(f"  Return: {result['metrics']['total_return']:.2%}")
            print(f"  Drawdown: {result['metrics']['max_drawdown']:.2%}")
            print(f"  Win Rate: {result['metrics']['win_rate']:.2%}")
            print(f"  Trades: {result['metrics']['trades_count']}")

        # Sla resultaten op
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True, parents=True)
        results_file = output_path / f"{strategy_name}_{symbol}_quick_optim.json"

        with open(results_file, 'w') as f:
            json.dump({
                'strategy': strategy_name,
                'symbol': symbol,
                'total_combinations': total_combinations,
                'top_results': [{
                    'params': result['params'],
                    'metrics': result['metrics']
                } for result in top_results]
            }, f, indent=2)

        print(f"\nResultaten opgeslagen in: {results_file}")

        # Test beste parameters nog een keer (simpele versie)
        best_params = top_results[0]['params']
        print(f"\n--- VERIFICATIE VAN BESTE PARAMETERS ---")

        try:
            df_verify = df_original.copy(deep=True)
            strategy = get_strategy(strategy_name, **best_params)

            entries, sl_stop, tp_stop = strategy.generate_signals(df_verify)

            # Converteer naar numpy arrays voor VectorBT
            entries_np = entries.values
            sl_stop_np = sl_stop.values
            tp_stop_np = tp_stop.values
            close_np = df_verify['close'].values

            pf = vbt.Portfolio.from_signals(
                close=close_np,
                entries=entries_np,
                sl_stop=sl_stop_np,
                tp_stop=tp_stop_np,
                init_cash=INITIAL_CAPITAL,
                fees=FEES,
                freq='1D'
            )

            print(f"Return: {pf.total_return():.2%}")
            print(f"Sharpe: {pf.sharpe_ratio():.2f}")
            print(f"Max Drawdown: {pf.max_drawdown():.2%}")
            print(f"Win Rate: {pf.trades.win_rate():.2%}")
            print(f"Trades: {len(pf.trades)}")

            # Maak equity curve plot
            plt.figure(figsize=(12, 6))
            pf.plot()
            plt.title(f"{strategy_name} Equity Curve (Geoptimaliseerde Parameters)")
            plt.savefig(output_path / f"{strategy_name}_{symbol}_best_equity.png")
            plt.close()

            print(f"Equity curve opgeslagen in: {output_path / f'{strategy_name}_{symbol}_best_equity.png'}")

            return top_results, pf

        except Exception as e:
            print(f"❌ Verificatie fout: {str(e)}")
    else:
        print("❌ Geen succesvolle resultaten gevonden.")

    return results, None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick Strategy Optimizer")
    parser.add_argument("--strategy", type=str, default="BollongStrategy", help="Naam van de strategie")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help=f"Trading symbool (default: {SYMBOL})")

    args = parser.parse_args()

    quick_optimize(args.strategy, args.symbol)