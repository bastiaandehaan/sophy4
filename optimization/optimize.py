# optimization/optimize.py
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import fetch_historical_data
from strategies import get_strategy, STRATEGIES
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from config import SYMBOL, INITIAL_CAPITAL, FEES, OUTPUT_DIR, logger

def optimize_strategy(strategy_name, symbol=SYMBOL, param_ranges=None, timeframe=None,
                      metric="sharpe_ratio", top_n=3, initial_capital=INITIAL_CAPITAL,
                      fees=FEES, period_days=1095, ftmo_compliant_only=False):
    start_time = time.time()
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)

    if strategy_name not in STRATEGIES:
        logger.error(f"Strategie '{strategy_name}' niet gevonden")
        return None

    strategy_class = STRATEGIES[strategy_name]
    param_ranges = param_ranges or strategy_class.get_default_params()

    df = fetch_historical_data(symbol, timeframe=timeframe, days=period_days)
    if df is None:
        logger.error(f"Geen data voor {symbol}")
        return None

    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] if isinstance(param_ranges[name], list) else [param_ranges[name]]
                    for name in param_names]
    param_combinations = list(product(*param_values))
    total_combinations = len(param_combinations)

    logger.info(f"Start {strategy_name} optimalisatie voor {symbol} met {total_combinations} combinaties")

    results = []
    processed = 0
    compliant_count = 0

    for params in param_combinations:
        param_dict = dict(zip(param_names, params))
        try:
            strategy = get_strategy(strategy_name, **param_dict)
            strategy.validate_parameters()

            # Genereer signalen en log eventuele NaN's ter controle
            entries, sl_stop, tp_stop = strategy.generate_signals(df.copy())
            logger.debug(f"NaN in entries: {entries.isna().any()}, NaN in sl_stop: {sl_stop.isna().any()}, "
                         f"NaN in tp_stop: {tp_stop.isna().any()}")

            # Converteer naar NumPy-arrays zonder .fillna(), want dat is al gedaan in generate_signals
            close_np = df['close'].values
            entries_np = np.asarray(entries, dtype=bool)
            sl_stop_np = np.asarray(sl_stop, dtype=float)
            tp_stop_np = np.asarray(tp_stop, dtype=float)

            # Verwijder NaN's uit alle arrays (veiligheidsmaatregel)
            mask = ~np.isnan(close_np) & ~np.isnan(entries_np) & ~np.isnan(sl_stop_np) & ~np.isnan(tp_stop_np)
            close_np = close_np[mask]
            entries_np = entries_np[mask]
            sl_stop_np = sl_stop_np[mask]
            tp_stop_np = tp_stop_np[mask]

            # Controleer of er voldoende data overblijft na maskeren
            if len(close_np) == 0:
                logger.warning(f"Geen geldige data na NaN-verwijdering voor {param_dict}")
                continue

            # Maak portfolio met vectorbt
            portfolio_kwargs = {
                'close': pd.Series(close_np, index=df.index[:len(close_np)]),
                'entries': pd.Series(entries_np, index=df.index[:len(entries_np)]),
                'sl_stop': pd.Series(sl_stop_np, index=df.index[:len(sl_stop_np)]),
                'tp_stop': pd.Series(tp_stop_np, index=df.index[:len(tp_stop_np)]),
                'init_cash': initial_capital,
                'fees': fees,
                'freq': '1D'
            }

            pf = vbt.Portfolio.from_signals(**portfolio_kwargs)

            # Bereken prestatiemetrics
            metrics_dict = {
                'total_return': float(pf.total_return()),
                'sharpe_ratio': float(pf.sharpe_ratio()),
                'sortino_ratio': float(pf.sortino_ratio()),
                'calmar_ratio': float(pf.calmar_ratio()),
                'max_drawdown': float(pf.max_drawdown()),
                'win_rate': float(pf.trades.win_rate() if len(pf.trades) > 0 else 0),
                'trades_count': len(pf.trades),
                'profit_factor': float(pf.trades['pnl'].sum() / abs(pf.trades['pnl'][pf.trades['pnl'] < 0].sum())
                                      if len(pf.trades[pf.trades['pnl'] < 0]) > 0 else float('inf'))
            }

            # Controleer FTMO-compliance
            compliant, profit_target = check_ftmo_compliance(pf, metrics_dict)
            if not ftmo_compliant_only or (compliant and profit_target):
                compliant_count += 1
                results.append({'params': param_dict, 'metrics': metrics_dict,
                               'ftmo_compliant': compliant, 'profit_target_reached': profit_target})

        except Exception as e:
            logger.warning(f"Fout bij {param_dict}: {str(e)}")
            continue

        processed += 1
        if processed % max(1, total_combinations // 20) == 0:
            elapsed = time.time() - start_time
            logger.info(f"Voortgang: {processed}/{total_combinations} ({processed / total_combinations * 100:.1f}%)")

    # Sorteer en selecteer topresultaten
    sorted_results = sorted(results, key=lambda x: x['metrics'][metric], reverse=metric != 'max_drawdown')
    top_results = sorted_results[:top_n]

    # Sla resultaten op
    timeframe_str = f"_{timeframe}" if timeframe else ""
    results_file = output_path / f"{strategy_name}_{symbol}{timeframe_str}_optimized.json"
    with open(results_file, 'w') as f:
        json.dump({'strategy': strategy_name, 'symbol': symbol, 'timeframe': str(timeframe),
                  'total_combinations': total_combinations, 'ftmo_compliant_count': compliant_count,
                  'top_results': [{'params': r['params'], 'metrics': r['metrics']} for r in top_results]},
                  f, indent=2)

    logger.info(f"Optimalisatie voltooid in {time.time() - start_time:.1f}s")
    return top_results

# De rest van de code blijft ongewijzigd
def walk_forward_test(strategy_name, symbol, params, period_days=1095, windows=5):
    logger.info(f"Walk-forward test voor {strategy_name} op {symbol}")
    df = fetch_historical_data(symbol, days=period_days)
    if df is None:
        logger.error(f"Geen data voor {symbol}")
        return None

    window_size = len(df) // windows
    results = []

    for i in range(windows - 1):
        train_start = i * window_size
        train_end = (i + 1) * window_size
        test_end = min(len(df), train_end + window_size)

        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        strategy = get_strategy(strategy_name, **params)
        train_entries, train_sl, train_tp = strategy.generate_signals(train_df)
        train_pf = vbt.Portfolio.from_signals(close=train_df['close'].values,
            entries=train_entries.values, sl_stop=train_sl.values,
            tp_stop=train_tp.values, init_cash=INITIAL_CAPITAL, fees=FEES, freq='1D')

        test_entries, test_sl, test_tp = strategy.generate_signals(test_df)
        test_pf = vbt.Portfolio.from_signals(close=test_df['close'].values,
            entries=test_entries.values, sl_stop=test_sl.values,
            tp_stop=test_tp.values, init_cash=INITIAL_CAPITAL, fees=FEES, freq='1D')

        window_result = {'window': i + 1,
            'train': {'return': float(train_pf.total_return()),
                'sharpe': float(train_pf.sharpe_ratio()),
                'max_drawdown': float(train_pf.max_drawdown()),
                'win_rate': float(train_pf.trades.win_rate() if len(train_pf.trades) > 0 else 0)},
            'test': {'return': float(test_pf.total_return()),
                'sharpe': float(test_pf.sharpe_ratio()),
                'max_drawdown': float(test_pf.max_drawdown()),
                'win_rate': float(test_pf.trades.win_rate() if len(test_pf.trades) > 0 else 0)}}
        results.append(window_result)

        logger.info(f"Window {i + 1}: Train Return {window_result['train']['return']:.2%}, "
                    f"Test Return {window_result['test']['return']:.2%}")

    avg_train_return = sum(r['train']['return'] for r in results) / len(results)
    avg_test_return = sum(r['test']['return'] for r in results) / len(results)

    logger.info(f"Walk-Forward Resultaat: Gem. Train Return {avg_train_return:.2%}, "
                f"Gem. Test Return {avg_test_return:.2%}")
    return results

def multi_instrument_test(strategy_name, parameters, instruments, timeframe=None):
    logger.info(f"Multi-instrument test voor {strategy_name} op {len(instruments)} instrumenten")
    results = {}

    for symbol in instruments:
        logger.info(f"Testen op {symbol}...")
        df = fetch_historical_data(symbol, timeframe=timeframe)
        if df is None:
            logger.warning(f"Geen data voor {symbol}, wordt overgeslagen")
            continue

        strategy = get_strategy(strategy_name, **parameters)
        entries, sl_stop, tp_stop = strategy.generate_signals(df)
        portfolio_kwargs = {
            'close': df['close'],
            'entries': entries,
            'sl_stop': sl_stop,
            'tp_stop': tp_stop,
            'init_cash': INITIAL_CAPITAL,
            'fees': FEES,
            'freq': '1D'
        }

        pf = vbt.Portfolio.from_signals(**portfolio_kwargs)

        metrics = {'total_return': float(pf.total_return()),
            'sharpe_ratio': float(pf.sharpe_ratio()),
            'max_drawdown': float(pf.max_drawdown()),
            'win_rate': float(pf.trades.win_rate() if len(pf.trades) > 0 else 0),
            'trades_count': len(pf.trades)}

        compliant, profit_target = check_ftmo_compliance(pf, metrics)
        results[symbol] = {'metrics': metrics, 'ftmo_compliant': compliant,
            'profit_target_reached': profit_target}

        logger.info(f"{symbol}: Return {metrics['total_return']:.2%}, "
                    f"Sharpe {metrics['sharpe_ratio']:.2f}, FTMO: {'JA' if compliant else 'NEE'}")

    output_path = Path(OUTPUT_DIR)
    timeframe_str = f"_{timeframe}" if timeframe else ""
    results_file = output_path / f"{strategy_name}_multi_instrument{timeframe_str}.json"
    with open(results_file, 'w') as f:
        json.dump({'strategy': strategy_name, 'parameters': parameters,
            'timeframe': str(timeframe), 'results': results}, f, indent=2)

    logger.info(f"Multi-instrument resultaten opgeslagen in {results_file}")
    return results