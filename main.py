#!/usr/bin/env python
"""
Sophy4 Trading Framework - Main script for backtesting, optimization, and analysis.
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import vectorbt as vbt
from backtest.backtest import run_extended_backtest
from config import logger, INITIAL_CAPITAL, SYMBOLS
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from monitor.monitor import monitor_performance
from strategies import STRATEGIES

def run_monte_carlo_simulation(pf, n_simulations=1000):
    """Monte Carlo simulation for portfolio performance using VectorBT."""
    if not hasattr(pf, 'trades') or len(pf.trades) < 10:
        return {
            'return_mean': 0.0,
            'return_95ci_lower': 0.0,
            'return_95ci_upper': 0.0,
            'profit_probability': 0.0,
            'drawdown_mean': 0.0,
            'drawdown_95ci': 0.0
        }

    trade_returns = pf.trades.returns
    final_returns = []
    max_drawdowns = []

    for _ in range(n_simulations):
        sim_returns = vbt.Portfolio.from_random_signals(
            close=pf.close,
            entries=pd.Series(np.random.choice([True, False], size=len(pf.close), p=[trade_returns.mean(), 1 - trade_returns.mean()])),
            size=pf.trades.size.mean(),
            init_cash=pf.init_cash,
            fees=pf.fees
        )
        final_returns.append(sim_returns.total_return())
        max_drawdowns.append(sim_returns.max_drawdown())

    final_returns = np.array(final_returns)
    max_drawdowns = np.array(max_drawdowns)

    return {
        'return_mean': np.mean(final_returns),
        'return_95ci_lower': np.percentile(final_returns, 2.5),
        'return_95ci_upper': np.percentile(final_returns, 97.5),
        'profit_probability': np.mean(final_returns > 0),
        'drawdown_mean': np.mean(max_drawdowns),
        'drawdown_95ci': np.percentile(max_drawdowns, 95)
    }

def run_walk_forward_test(strategy_name: str, parameters: Dict[str, Any], symbol: str,
                          timeframe: str, total_days: int, windows: int = 3,
                          test_percent: float = 0.3) -> Dict[str, Any]:
    """Perform a walk-forward test with dynamic parameter optimization."""
    logger.info(f"Starting walk-forward test: {strategy_name} on {symbol}")

    window_days = total_days // windows
    train_days = int(window_days * (1 - test_percent))
    test_days = window_days - train_days

    results = {
        'windows_tested': 0,
        'window_results': [],
        'avg_train_return': 0.0,
        'avg_test_return': 0.0,
        'avg_train_sharpe': 0.0,
        'avg_test_sharpe': 0.0,
        'robustness': {
            'return_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'is_robust': False
        }
    }

    train_returns, test_returns, train_sharpes, test_sharpes = [], [], [], []
    end_date = datetime.now()

    for i in range(windows):
        window_end_date = end_date - timedelta(days=i * window_days)
        test_start_date = window_end_date - timedelta(days=test_days)
        train_start_date = test_start_date - timedelta(days=train_days)

        # Optimize parameters on training data
        logger.info(f"Window {i+1}: Optimizing parameters on training data")
        optimizer = vbt.ParameterOptimizer(
            strategy_name=strategy_name,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            period_days=train_days,
            end_date=test_start_date
        )
        best_params = optimizer.optimize(metric='sharpe_ratio')

        # Run backtest on training data
        logger.info(f"Window {i+1}: Training from {train_start_date.date()} to {test_start_date.date()}")
        train_pf, train_metrics = run_extended_backtest(
            strategy_name=strategy_name,
            parameters=best_params,
            symbol=symbol,
            timeframe=timeframe,
            period_days=train_days,
            end_date=test_start_date
        )

        # Run backtest on test data
        logger.info(f"Window {i+1}: Testing from {test_start_date.date()} to {window_end_date.date()}")
        test_pf, test_metrics = run_extended_backtest(
            strategy_name=strategy_name,
            parameters=best_params,
            symbol=symbol,
            timeframe=timeframe,
            period_days=test_days,
            end_date=window_end_date
        )

        if train_pf is None or test_pf is None:
            logger.warning(f"Window {i+1}: No valid results, skipping this window")
            continue

        train_return = train_metrics.get('total_return', 0.0)
        test_return = test_metrics.get('total_return', 0.0)
        train_sharpe = train_metrics.get('sharpe_ratio', 0.0)
        test_sharpe = test_metrics.get('sharpe_ratio', 0.0)

        train_returns.append(train_return)
        test_returns.append(test_return)
        train_sharpes.append(train_sharpe)
        test_sharpes.append(test_sharpe)

        results['window_results'].append({
            'window': i+1,
            'train_period': {
                'start': train_start_date.strftime('%Y-%m-%d'),
                'end': test_start_date.strftime('%Y-%m-%d'),
                'return': train_return,
                'sharpe': train_sharpe
            },
            'test_period': {
                'start': test_start_date.strftime('%Y-%m-%d'),
                'end': window_end_date.strftime('%Y-%m-%d'),
                'return': test_return,
                'sharpe': test_sharpe
            }
        })
        results['windows_tested'] += 1

    if results['windows_tested'] > 0:
        results['avg_train_return'] = sum(train_returns) / len(train_returns)
        results['avg_test_return'] = sum(test_returns) / len(test_returns)
        results['avg_train_sharpe'] = sum(train_sharpes) / len(train_sharpes)
        results['avg_test_sharpe'] = sum(test_sharpes) / len(test_sharpes)

        return_ratio = results['avg_test_return'] / results['avg_train_return'] if results['avg_train_return'] > 0 else 0
        sharpe_ratio = results['avg_test_sharpe'] / results['avg_train_sharpe'] if results['avg_train_sharpe'] > 0 else 0
        results['robustness'] = {
            'return_ratio': return_ratio,
            'sharpe_ratio': sharpe_ratio,
            'is_robust': return_ratio >= 0.7 and sharpe_ratio >= 0.7
        }

    return results

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy4 Trading Framework")
    parser.add_argument("--mode", type=str, choices=["backtest", "optimize", "monte_carlo", "walk_forward"],
                        default="backtest", help="Operation mode")
    parser.add_argument("--symbol", type=str, default=SYMBOLS[0], help=f"Trading symbol (default: {SYMBOLS[0]})")
    parser.add_argument("--strategy", type=str, default="OrderBlockLSTMStrategy", help="Strategy to use")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (e.g., M15, H1, D1)")
    parser.add_argument("--days", type=int, default=1095, help="Number of days of historical data")
    parser.add_argument("--params_file", type=str, help="JSON file with parameters")
    parser.add_argument("--params_index", type=int, default=0, help="Index in the params file (0 = best)")
    parser.add_argument("--metric", type=str, default="sharpe_ratio", help="Metric for optimization")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top results to show")
    parser.add_argument("--quick", action="store_true", help="Quick optimization with fewer parameters")
    parser.add_argument("--windows", type=int, default=3, help="Number of windows for walk-forward test")
    parser.add_argument("--test_pct", type=float, default=0.3, help="Percentage of window for testing")
    parser.add_argument("--window", type=int, help="Window period")
    parser.add_argument("--std_dev", type=float, help="Standard deviation")
    parser.add_argument("--sl_fixed_percent", type=float, help="Stop-loss percentage")
    parser.add_argument("--tp_fixed_percent", type=float, help="Take-profit percentage")
    parser.add_argument("--risk_per_trade", type=float, default=0.01, help="Risk per trade")
    parser.add_argument("--use_trailing_stop", action="store_true", help="Use trailing stop")
    parser.add_argument("--trailing_stop_percent", type=float, help="Trailing stop percentage")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="VaR confidence level")
    parser.add_argument("--model_path", type=str, help="Path to trained LSTM model (.h5 file)")
    parser.add_argument("--time_filter_pct", type=float, help="Percentage of time for trading filter")
    parser.add_argument("--verbose_logging", action="store_true", help="Show detailed logging information")
    return parser.parse_args()

def load_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    """Load parameters from command line or params_file."""
    parameters = {}
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > args.params_index:
                    parameters = data[args.params_index]['params']
                    logger.info(f"Parameters loaded from {args.params_file} (index {args.params_index})")
        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}")

    direct_params = ['window', 'std_dev', 'sl_fixed_percent', 'tp_fixed_percent',
                     'risk_per_trade', 'use_trailing_stop', 'trailing_stop_percent',
                     'confidence_level', 'model_path', 'time_filter_pct']
    for param in direct_params:
        value = getattr(args, param, None)
        if value is not None:
            parameters[param] = value

    parameters['symbol'] = args.symbol
    return parameters

def save_results_to_json(strategy: str, symbol: str, results: Dict[str, Any]) -> None:
    """Save results to a JSON file with a timestamp."""
    output_path = Path("results")
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    with open(output_path / f"{strategy}_{symbol}_walkforward_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.strategy not in STRATEGIES:
        logger.error(f"Strategy '{args.strategy}' not found. Available strategies: {', '.join(STRATEGIES.keys())}")
        return

    if args.symbol not in SYMBOLS:
        logger.error(f"Symbol '{args.symbol}' not in configured SYMBOLS: {SYMBOLS}")
        return

    if args.mode == "backtest":
        parameters = load_parameters(args)
        if not parameters:
            logger.error("No parameters specified for backtest")
            return

        pf, metrics = run_extended_backtest(
            strategy_name=args.strategy,
            parameters=parameters,
            symbol=args.symbol,
            timeframe=args.timeframe,
            period_days=args.days,
            initial_capital=INITIAL_CAPITAL
        )
        if pf is not None:
            compliant, profit_target = check_ftmo_compliance(pf, metrics)
            monitor_performance(pf)

    elif args.mode == "monte_carlo":
        parameters = load_parameters(args)
        if not parameters:
            logger.error("No parameters specified for Monte Carlo")
            return

        pf, _ = run_extended_backtest(
            strategy_name=args.strategy,
            parameters=parameters,
            symbol=args.symbol,
            timeframe=args.timeframe,
            period_days=args.days
        )
        if pf is not None and len(pf.trades) >= 10:
            mc_results = run_monte_carlo_simulation(pf, n_simulations=1000)
            print("\n=== MONTE CARLO RESULTS ===")
            print(f"Average Return: {mc_results['return_mean']:.2%}")
            print(f"95% Confidence Interval: [{mc_results['return_95ci_lower']:.2%} to {mc_results['return_95ci_upper']:.2%}]")
            print(f"Profit Probability: {mc_results['profit_probability']:.2%}")
            print(f"Average Max Drawdown: {mc_results['drawdown_mean']:.2%}")
            print(f"Worst-Case Drawdown (95%): {mc_results['drawdown_95ci']:.2%}")
        else:
            logger.error("Insufficient trades for Monte Carlo simulation")

    elif args.mode == "optimize":
        parameters = load_parameters(args)
        optimizer = vbt.ParameterOptimizer(
            strategy_name=args.strategy,
            parameters=parameters,
            symbol=args.symbol,
            timeframe=args.timeframe,
            period_days=args.days
        )
        best_params = optimizer.optimize(metric=args.metric, quick=args.quick)
        logger.info(f"Best parameters: {best_params}")

    elif args.mode == "walk_forward":
        parameters = load_parameters(args)
        if not parameters:
            logger.error("No parameters specified for walk-forward test")
            return

        results = run_walk_forward_test(
            strategy_name=args.strategy,
            parameters=parameters,
            symbol=args.symbol,
            timeframe=args.timeframe,
            total_days=args.days,
            test_percent=args.test_pct,
            windows=args.windows
        )
        if results:
            print("\n=== WALK-FORWARD TEST RESULTS ===")
            print(f"Windows Tested: {results['windows_tested']}")
            print(f"Average Train Return: {results['avg_train_return']:.2%}")
            print(f"Average Test Return: {results['avg_test_return']:.2%}")
            print(f"Average Train Sharpe: {results['avg_train_sharpe']:.2f}")
            print(f"Average Test Sharpe: {results['avg_test_sharpe']:.2f}")
            print(f"Return Ratio (Test/Train): {results['robustness']['return_ratio']:.2f}")
            print(f"Sharpe Ratio (Test/Train): {results['robustness']['sharpe_ratio']:.2f}")
            print(f"Strategy is {'ROBUST' if results['robustness']['is_robust'] else 'NOT ROBUST'}")
            save_results_to_json(args.strategy, args.symbol, results)

if __name__ == "__main__":
    main()