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
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
from tqdm import tqdm

# Voeg projectroot toe aan Python path
sys.path.append(str(Path(__file__).parent.parent))

from backtest.backtest import run_extended_backtest
from config import logger, OUTPUT_DIR
from risk.risk_management import RiskManager
from strategies import get_strategy, STRATEGIES
from utils.data_utils import fetch_historical_data

# Maak een summary logger
summary_logger = logging.getLogger("summary")
summary_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
if not any(isinstance(h, logging.StreamHandler) for h in summary_logger.handlers):
    summary_logger.addHandler(console_handler)
summary_logger.propagate = False

def quick_optimize(strategy_name: str, symbol: str, timeframe: str = "D1", days: int = 365,
                   metric: str = "sharpe_ratio", top_n: int = 5, verbose: bool = False,
                   quick: bool = False) -> List[Dict[str, Any]]:
    """
    Voert een snelle parameter optimalisatie uit en slaat de resultaten op.

    Args:
        strategy_name: Naam van de strategie
        symbol: Handelssymbool
        timeframe: Timeframe ("D1", "H4", etc)
        days: Aantal dagen historische data
        metric: Metric om op te optimaliseren
        top_n: Aantal beste resultaten om te tonen
        verbose: Toon uitgebreide logs
        quick: Gebruik een beperkte parameter grid (sneller)

    Returns:
        Lijst met top resultaten
    """
    start_time: float = time.time()

    if verbose:
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)

    summary_logger.info(f"\n{'=' * 80}")
    summary_logger.info(f"OPTIMALISATIE GESTART: {strategy_name} op {symbol} ({timeframe})")
    summary_logger.info(f"{'=' * 80}")

    if strategy_name not in STRATEGIES:
        available: str = ", ".join(STRATEGIES.keys())
        summary_logger.error(f"Strategie {strategy_name} niet gevonden. Beschikbaar: {available}")
        return []

    strategy_class = STRATEGIES[strategy_name]
    param_grid: Dict[str, List[Any]] = strategy_class.get_default_params(timeframe=timeframe)
    param_grid['confidence_level'] = [0.90, 0.95, 0.99]  # VaR-parameter

    if quick:
        for key, values in param_grid.items():
            if isinstance(values, list) and len(values) > 3:
                param_grid[key] = [values[0], values[len(values) // 2], values[-1]]

    param_names: List[str] = list(param_grid.keys())
    param_values: List[List[Any]] = list(param_grid.values())
    param_combinations: List[Tuple[Any, ...]] = list(itertools.product(*param_values))
    total_combinations: int = len(param_combinations)

    summary_logger.info(f"Testen van {total_combinations} parameter combinaties...")

    df: Optional[pd.DataFrame] = fetch_historical_data(symbol, timeframe=timeframe, days=days)
    if df is None or df.empty:
        summary_logger.error(f"Geen data beschikbaar voor {symbol}")
        return []

    results: List[Dict[str, Any]] = []
    result_hash: set = set()

    summary_logger.info(f"Voortgang:")
    for values in tqdm(param_combinations, desc="Parameter combinaties", unit="test"):
        params: Dict[str, Any] = dict(zip(param_names, values))

        try:
            risk_manager: RiskManager = RiskManager(confidence_level=params.get('confidence_level', 0.95),
                                                    max_risk=params.get('risk_per_trade', 0.01))
            returns: pd.Series = df['close'].pct_change().dropna()
            size: float = risk_manager.calculate_position_size(10000, returns, pip_value=10.0)

            strategy = get_strategy(strategy_name, **params)
            entries, sl_stop, tp_stop = strategy.generate_signals(df)
            if entries.sum() == 0:
                continue

            pf, metrics = run_extended_backtest(strategy_name, params, symbol, timeframe, days)
            if not metrics:
                continue

            if metric not in metrics:
                metric = list(metrics.keys())[0]
                if verbose:
                    summary_logger.warning(f"Metric {metric} niet gevonden, gebruik {metric}")

            metrics['signal_count'] = int(entries.sum())
            metrics['trade_count'] = metrics.get('trades_count', 0)

            result_key: Tuple[float, float, float, int] = (
                round(metrics.get('sharpe_ratio', 0), 3),
                round(metrics.get('total_return', 0), 3),
                round(metrics.get('max_drawdown', 0), 3),
                metrics.get('trade_count', 0)
            )
            if result_key not in result_hash:
                result_hash.add(result_key)
                results.append({'params': params, 'metrics': metrics})

        except Exception as e:
            if verbose:
                logger.error(f"Fout bij testen parameters {params}: {str(e)}")

    if results:
        sorted_results: List[Dict[str, Any]] = sorted(results, key=lambda x: x['metrics'].get(metric, -999999), reverse=True)
        top_results: List[Dict[str, Any]] = sorted_results[:top_n]

        summary_logger.info(f"\n{'=' * 80}")
        summary_logger.info(f"TOP {len(top_results)} RESULTATEN VOOR {strategy_name} OP {symbol}:")
        summary_logger.info(f"{'=' * 80}")
        summary_logger.info(f"{'#':<3} {'Sharpe':<8} {'Return':<8} {'DrawDn':<8} {'WinRate':<8} {'Trades':<6} {'Params'}")
        summary_logger.info(f"{'-' * 80}")

        for i, result in enumerate(top_results):
            metrics = result['metrics']
            params_str: str = ", ".join([f"{k}={v}" for k, v in result['params'].items() if
                                        k in ['window', 'std_dev', 'sl_fixed_percent', 'tp_fixed_percent',
                                              'use_trailing_stop', 'trailing_stop_percent', 'confidence_level']])
            summary_logger.info(f"{i + 1:<3} {metrics.get('sharpe_ratio', 0):.2f}     "
                                f"{metrics.get('total_return', 0) * 100:.2f}%    "
                                f"{metrics.get('max_drawdown', 0) * 100:.2f}%    "
                                f"{metrics.get('win_rate', 0) * 100:.2f}%    "
                                f"{metrics.get('trade_count', 0):<6} {params_str}")

        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M")
        output_path: Path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True, parents=True)
        output_file: Path = output_path / f"{strategy_name}_{symbol}_quick_optim_{timestamp}.json"
        with open(output_file, 'w') as f:
            serializable_results: List[Dict[str, Any]] = [
                {'params': r['params'], 'metrics': {k: float(v) if isinstance(v, (float, int)) else str(v)
                                                    for k, v in r['metrics'].items()}}
                for r in top_results
            ]
            json.dump(serializable_results, f, indent=2)

        elapsed_time: float = time.time() - start_time
        summary_logger.info(f"\nResultaten opgeslagen in {output_file}")
        summary_logger.info(f"Optimalisatie voltooid in {elapsed_time:.1f} seconden.")
        return top_results

    summary_logger.warning("\nGeen resultaten gevonden")
    return []

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy4 Quick Optimization Tool")
    parser.add_argument("--strategy", type=str, required=True, help="Strategie om te optimaliseren")
    parser.add_argument("--symbol", type=str, default="GER40.cash", help="Handelssymbool")
    parser.add_argument("--timeframe", type=str, default="D1", help="Timeframe")
    parser.add_argument("--days", type=int, default=365, help="Aantal dagen historische data")
    parser.add_argument("--metric", type=str, default="sharpe_ratio", help="Metric om te optimaliseren")
    parser.add_argument("--top_n", type=int, default=5, help="Aantal beste parameter sets")
    parser.add_argument("--verbose", action="store_true", help="Toon alle log berichten")
    parser.add_argument("--quick", action="store_true", help="Gebruik een beperkte parameter grid (sneller)")
    return parser.parse_args()

if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    quick_optimize(args.strategy, args.symbol, args.timeframe, args.days, args.metric,
                   args.top_n, args.verbose, args.quick)