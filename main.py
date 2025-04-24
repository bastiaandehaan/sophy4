#!/usr/bin/env python
# main.py
"""
Sophy4 Trading Framework - Hoofdscript voor backtest, optimalisatie en analyse
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Wijzig de import om alleen de functie te importeren die wel bestaat
from backtest.backtest import run_extended_backtest
from config import logger, INITIAL_CAPITAL, SYMBOLS
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from monitor.monitor import monitor_performance
from optimization.optimize import quick_optimize
from strategies import STRATEGIES

# Implementatie van Monte Carlo simulatie functie
def run_monte_carlo_simulation(pf, n_simulations=1000):
    """
    Monte Carlo simulatie voor portfolio performance.

    Args:
        pf: Portfolio met trading history
        n_simulations: Aantal simulaties

    Returns:
        Dict met simulatie resultaten
    """
    import numpy as np
    import random

    # Controleer of we trade data hebben
    if not hasattr(pf, 'trades') or len(pf.trades) < 10:
        return {
            'return_mean': 0.0,
            'return_95ci_lower': 0.0,
            'return_95ci_upper': 0.0,
            'profit_probability': 0.0,
            'drawdown_mean': 0.0,
            'drawdown_95ci': 0.0
        }

    # Extraheer trade returns als percentages
    trade_returns = [trade.pnl_pct for trade in pf.trades if hasattr(trade, 'pnl_pct')]

    # Voer simulaties uit
    final_returns = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Willekeurig sampling van trades met replacement
        sim_returns = random.choices(trade_returns, k=len(trade_returns))

        # Bereken cumulatief return pad
        cum_returns = np.cumprod(1 + np.array(sim_returns)) - 1

        # Sla final return op
        final_returns.append(cum_returns[-1] if len(cum_returns) > 0 else 0)

        # Bereken max drawdown
        peak = 0
        max_dd = 0
        for ret in cum_returns:
            if ret > peak:
                peak = ret
            dd = (peak - ret) / (1 + peak) if peak > 0 else 0
            max_dd = max(max_dd, dd)
        max_drawdowns.append(max_dd)

    # Bereken statistieken
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


# Implementatie van Walk Forward Test functie
def run_walk_forward_test(strategy_name: str, parameters: Dict[str, Any], symbol: str,
                          timeframe: str, total_days: int, windows: int = 3,
                          test_percent: float = 0.3) -> Dict[str, Any]:
    """
    Voert een walk-forward test uit voor een strategie.

    Args:
        strategy_name: Naam van de strategie
        parameters: Parameters voor de strategie
        symbol: Handelssymbool
        timeframe: Timeframe voor de data
        total_days: Totaal aantal dagen voor de test
        windows: Aantal windows om te testen
        test_percent: Percentage van elke window voor testing

    Returns:
        Dict met resultaten van de walk-forward test
    """
    logger.info(f"Start walk-forward test: {strategy_name} op {symbol}")

    # Bereken window grootte in dagen
    window_days = total_days // windows
    train_days = int(window_days * (1 - test_percent))
    test_days = window_days - train_days

    # Resultaten opslag
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

    train_returns = []
    test_returns = []
    train_sharpes = []
    test_sharpes = []

    # Huidige datum voor het bepalen van eindpunten
    end_date = datetime.now()

    for i in range(windows):
        # Bereken datums voor deze window
        window_end_date = end_date - timedelta(days=i * window_days)
        test_start_date = window_end_date - timedelta(days=test_days)
        train_start_date = test_start_date - timedelta(days=train_days)

        # Train periode backtest
        logger.info(f"Window {i+1}: Training van {train_start_date.date()} tot {test_start_date.date()}")
        train_pf, train_metrics = run_extended_backtest(
            strategy_name=strategy_name,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            period_days=train_days,
            end_date=test_start_date
        )

        # Test periode backtest
        logger.info(f"Window {i+1}: Testing van {test_start_date.date()} tot {window_end_date.date()}")
        test_pf, test_metrics = run_extended_backtest(
            strategy_name=strategy_name,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            period_days=test_days,
            end_date=window_end_date
        )

        # Controleer of beide backtests resultaten hebben opgeleverd
        if train_pf is None or test_pf is None:
            logger.warning(f"Window {i+1}: Geen geldige resultaten, deze window wordt overgeslagen")
            continue

        # Verzamel resultaten
        train_return = train_metrics.get('total_return', 0.0)
        test_return = test_metrics.get('total_return', 0.0)
        train_sharpe = train_metrics.get('sharpe_ratio', 0.0)
        test_sharpe = test_metrics.get('sharpe_ratio', 0.0)

        # Voeg toe aan lijsten
        train_returns.append(train_return)
        test_returns.append(test_return)
        train_sharpes.append(train_sharpe)
        test_sharpes.append(test_sharpe)

        # Voeg resultaten toe aan window_results
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

        # Teller bijwerken
        results['windows_tested'] += 1

    # Bereken gemiddelden als er windows zijn getest
    if results['windows_tested'] > 0:
        results['avg_train_return'] = sum(train_returns) / len(train_returns)
        results['avg_test_return'] = sum(test_returns) / len(test_returns)
        results['avg_train_sharpe'] = sum(train_sharpes) / len(train_sharpes)
        results['avg_test_sharpe'] = sum(test_sharpes) / len(test_sharpes)

        # Bereken robuustheid metrics
        return_ratio = results['avg_test_return'] / results['avg_train_return'] if results['avg_train_return'] > 0 else 0
        sharpe_ratio = results['avg_test_sharpe'] / results['avg_train_sharpe'] if results['avg_train_sharpe'] > 0 else 0

        results['robustness'] = {
            'return_ratio': return_ratio,
            'sharpe_ratio': sharpe_ratio,
            'is_robust': return_ratio >= 0.7 and sharpe_ratio >= 0.7  # 70% behoud van performance
        }

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy4 Trading Framework")
    parser.add_argument("--mode", type=str, choices=["backtest", "optimize", "monte_carlo", "walk_forward"],
                        default="backtest", help="Operatie modus")
    parser.add_argument("--symbol", type=str, default=SYMBOLS[0], help=f"Trading symbool (default: {SYMBOLS[0]})")
    parser.add_argument("--strategy", type=str, default="OrderBlockLSTMStrategy", help="Strategie om te gebruiken")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (bijv. M15, H1, D1)")
    parser.add_argument("--days", type=int, default=1095, help="Aantal dagen historische data")
    parser.add_argument("--params_file", type=str, help="JSON bestand met parameters")
    parser.add_argument("--params_index", type=int, default=0, help="Index in het params bestand (0 = beste)")
    parser.add_argument("--metric", type=str, default="sharpe_ratio", help="Metric voor optimalisatie")
    parser.add_argument("--top_n", type=int, default=5, help="Aantal beste resultaten om te tonen")
    parser.add_argument("--quick", action="store_true", help="Snelle optimalisatie met minder parameters")
    parser.add_argument("--windows", type=int, default=3, help="Aantal windows voor walk-forward test")
    parser.add_argument("--test_pct", type=float, default=0.3, help="Percentage van window voor testing")
    parser.add_argument("--window", type=int, help="Window periode")
    parser.add_argument("--std_dev", type=float, help="Standaarddeviatie")
    parser.add_argument("--sl_fixed_percent", type=float, help="Stop-loss percentage")
    parser.add_argument("--tp_fixed_percent", type=float, help="Take-profit percentage")
    parser.add_argument("--risk_per_trade", type=float, default=0.01, help="Risico per trade")
    parser.add_argument("--use_trailing_stop", action="store_true", help="Gebruik trailing stop")
    parser.add_argument("--trailing_stop_percent", type=float, help="Trailing stop percentage")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="VaR confidence level")
    parser.add_argument("--model_path", type=str, help="Pad naar getraind LSTM model (.h5 bestand)")
    parser.add_argument("--time_filter_pct", type=float, help="Percentage van tijd voor trading filter")
    return parser.parse_args()


def load_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    """Laad parameters van commandline of params_file."""
    parameters: Dict[str, Any] = {}

    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                data: List[Dict[str, Any]] = json.load(f)
                if isinstance(data, list) and len(data) > args.params_index:
                    parameters = data[args.params_index]['params']
                    logger.info(f"Parameters geladen uit {args.params_file} (index {args.params_index})")
                else:
                    logger.warning(f"Ongeldig formaat in {args.params_file}")
        except Exception as e:
            logger.error(f"Fout bij laden parameters: {str(e)}")

    direct_params: List[str] = ['window', 'std_dev', 'sl_fixed_percent', 'tp_fixed_percent',
                                'risk_per_trade', 'use_trailing_stop', 'trailing_stop_percent',
                                'confidence_level', 'model_path', 'time_filter_pct']
    for param in direct_params:
        value = getattr(args, param, None)
        if value is not None:
            parameters[param] = value

    # Voeg symbol expliciet toe aan parameters
    parameters['symbol'] = args.symbol

    return parameters


def save_results_to_json(strategy: str, symbol: str, results: Dict[str, Any]) -> None:
    """
    Save results to a JSON file with a timestamp.

    Args:
        strategy (str): Name of the strategy.
        symbol (str): Trading symbol.
        results (Dict[str, Any]): Results to save.
    """
    output_path: Path = Path("results")
    output_path.mkdir(exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M")
    with open(output_path / f"{strategy}_{symbol}_walkforward_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


def main() -> None:
    """Main entry point."""
    args: argparse.Namespace = parse_args()

    if args.strategy not in STRATEGIES:
        available_strategies: str = ", ".join(STRATEGIES.keys())
        logger.error(f"Strategie '{args.strategy}' niet gevonden. Beschikbare strategieën: {available_strategies}")
        return

    if args.symbol not in SYMBOLS:
        logger.error(f"Symbool '{args.symbol}' niet in geconfigureerde SYMBOLS: {SYMBOLS}")
        return

    if args.mode == "backtest":
        parameters: Dict[str, Any] = load_parameters(args)
        if not parameters:
            logger.error("Geen parameters opgegeven voor backtest")
            return

        logger.info(f"Start backtest: {args.strategy} op {args.symbol}")
        logger.info(f"Parameters: {parameters}")

        pf, metrics = run_extended_backtest(strategy_name=args.strategy, parameters=parameters,
                                            symbol=args.symbol, timeframe=args.timeframe,
                                            period_days=args.days, initial_capital=INITIAL_CAPITAL)
        if pf is not None:
            compliant, profit_target = check_ftmo_compliance(pf, metrics)
            monitor_performance(pf)

            print("\n=== BACKTEST RESULTATEN ===")
            print(f"Totaal rendement: {metrics['total_return']:.2%}")
            print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Win rate: {metrics['win_rate']:.2%}")
            print(f"Aantal trades: {metrics['trades_count']}")
            print(f"FTMO compliant: {'JA' if compliant else 'NEE'}")
            print(f"Profit target bereikt: {'JA' if profit_target else 'NEE'}")
        else:
            logger.error("Backtest failed to produce results.")
            print("Backtest failed. Check logs for details.")

    elif args.mode == "monte_carlo":
        parameters: Dict[str, Any] = load_parameters(args)
        if not parameters:
            logger.error("Geen parameters opgegeven voor Monte Carlo")
            return

        logger.info(f"Start Monte Carlo: {args.strategy} op {args.symbol}")
        pf, _ = run_extended_backtest(strategy_name=args.strategy, parameters=parameters,
                                      symbol=args.symbol, timeframe=args.timeframe, period_days=args.days)
        if pf is not None and len(pf.trades) >= 10:
            mc_results: Dict[str, Any] = run_monte_carlo_simulation(pf, n_simulations=1000)
            print("\n=== MONTE CARLO RESULTATEN ===")
            print(f"Gemiddeld rendement: {mc_results['return_mean']:.2%}")
            print(f"95% betrouwbaarheidsinterval rendement: [{mc_results['return_95ci_lower']:.2%} tot {mc_results['return_95ci_upper']:.2%}]")
            print(f"Kans op winstgevendheid: {mc_results['profit_probability']:.2%}")
            print(f"Gemiddelde max drawdown: {mc_results['drawdown_mean']:.2%}")
            print(f"Worst-case drawdown (95%): {mc_results['drawdown_95ci']:.2%}")
        else:
            logger.error("Onvoldoende trades voor Monte Carlo simulatie")
            print("Onvoldoende trades voor Monte Carlo simulatie. Minimaal 10 trades nodig.")

    elif args.mode == "optimize":
        logger.info(f"Start optimalisatie: {args.strategy} op {args.symbol}")
        quick_optimize(strategy_name=args.strategy, symbol=args.symbol, timeframe=args.timeframe,
                       days=args.days, metric=args.metric, top_n=args.top_n, verbose=True, quick=args.quick)

    elif args.mode == "walk_forward":
        parameters: Dict[str, Any] = load_parameters(args)
        if not parameters:
            logger.error("Geen parameters opgegeven voor walk-forward test")
            return

        logger.info(f"Start walk-forward test: {args.strategy} op {args.symbol}")
        logger.info(f"Parameters: {parameters}")
        results: Dict[str, Any] = run_walk_forward_test(strategy_name=args.strategy, parameters=parameters,
                                                        symbol=args.symbol, timeframe=args.timeframe,
                                                        total_days=args.days, test_percent=args.test_pct,
                                                        windows=args.windows)
        if results:
            print("\n=== WALK-FORWARD TEST RESULTATEN ===")
            print(f"Windows getest: {results['windows_tested']}")
            print(f"Gemiddeld rendement Train: {results['avg_train_return']:.2%}")
            print(f"Gemiddeld rendement Test: {results['avg_test_return']:.2%}")
            print(f"Gemiddelde Sharpe Train: {results['avg_train_sharpe']:.2f}")
            print(f"Gemiddelde Sharpe Test: {results['avg_test_sharpe']:.2f}")
            print(f"Rendement ratio (Test/Train): {results['robustness']['return_ratio']:.2f}")
            print(f"Sharpe ratio (Test/Train): {results['robustness']['sharpe_ratio']:.2f}")
            print(f"Strategie is {'ROBUUST' if results['robustness']['is_robust'] else 'NIET ROBUUST'}")

            save_results_to_json(args.strategy, args.symbol, results)
        else:
            logger.error("Walk-forward test failed to produce results.")
            print("Walk-forward test failed. Check logs for details.")


if __name__ == "__main__":
    main()