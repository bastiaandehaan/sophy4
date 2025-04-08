# sophy_commander.py
import argparse
import json
from pathlib import Path

import MetaTrader5 as mt5

from backtest.extended_backtest import run_extended_backtest, monte_carlo_analysis
from config import SYMBOL, logger, OUTPUT_DIR
from optimization.optimize import optimize_strategy, walk_forward_test, \
    multi_instrument_test


def run_full_analysis(strategy, symbols=None, timeframes=None, monte_carlo=False):
    """
    Voer een complete analyse uit inclusief optimalisatie, walk-forward test,
    multi-instrument test en gedetailleerde backtests.
    """
    if symbols is None:
        symbols = [SYMBOL, 'US30.cash', 'EURUSD', 'GBPUSD']

    if timeframes is None:
        timeframes = {'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1}

    results = {}

    # 1. Voor elk symbool en timeframe, vind de beste parameters
    for symbol in symbols:
        symbol_results = {}

        for tf_name, tf_value in timeframes.items():
            logger.info(f"=== ANALYSE {strategy} op {symbol} ({tf_name}) ===")

            # Optimalisatie
            logger.info("1. Parameter optimalisatie...")
            optim_results = optimize_strategy(strategy_name=strategy, symbol=symbol,
                timeframe=tf_value, metric="sharpe_ratio", top_n=3,
                ftmo_compliant_only=True)

            if not optim_results:
                logger.error(f"Geen optimalisatie resultaten voor {symbol} {tf_name}")
                continue

            # Beste parameters
            best_params = optim_results[0]['params']
            logger.info(f"Beste parameters: {best_params}")

            # Walk-forward test
            logger.info("2. Walk-forward validatie...")
            wf_results = walk_forward_test(strategy_name=strategy, symbol=symbol,
                params=best_params, period_days=1095)

            # Gedetailleerde backtest
            logger.info("3. Uitgebreide backtest...")
            pf, backtest_metrics = run_extended_backtest(strategy_name=strategy,
                parameters=best_params, symbol=symbol, timeframe=tf_value)

            # Monte Carlo analyse
            mc_results = None
            if monte_carlo and pf is not None and backtest_metrics[
                'trades_count'] >= 10:
                logger.info("4. Monte Carlo analyse...")
                mc_results = monte_carlo_analysis(pf, n_simulations=1000)

            # Bewaar resultaten
            symbol_results[tf_name] = {'parameters': best_params,
                'metrics': backtest_metrics, 'walk_forward': wf_results,
                'monte_carlo': mc_results}

        # Bewaar resultaten per symbool
        results[symbol] = symbol_results

    # Vergelijk resultaten tussen instrumenten
    logger.info("\n=== VERGELIJKING TUSSEN INSTRUMENTEN ===")

    for tf_name in timeframes.keys():
        logger.info(f"\nTimeframe: {tf_name}")
        logger.info("Instrument   | Return   | Sharpe | Drawdown | Win Rate | Trades")
        logger.info("-------------|----------|--------|----------|----------|-------")

        for symbol in symbols:
            if symbol in results and tf_name in results[symbol]:
                metrics = results[symbol][tf_name]['metrics']
                logger.info(
                    f"{symbol:<12} | {metrics['total_return']:7.2%} | {metrics['sharpe_ratio']:6.2f} | {metrics['max_drawdown']:8.2%} | {metrics['win_rate']:7.2%} | {metrics['trades_count']:6}")

    # Sla samenvatting op
    output_path = Path(OUTPUT_DIR)
    summary_file = output_path / f"{strategy}_complete_analysis.json"

    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nComplete analyse opgeslagen in {summary_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sophy4 Commander - Complete Analyse Tool")

    parser.add_argument("--strategy", type=str, default="BollongStrategy",
                        help="Strategie om te analyseren (default: BollongStrategy)")
    parser.add_argument("--symbols", type=str, nargs="+",
                        default=[SYMBOL, 'US30.cash', 'EURUSD', 'GBPUSD'],
                        help="Lijst van instrumenten om te testen")
    parser.add_argument("--timeframes", type=str, nargs="+", default=['H1', 'H4', 'D1'],
                        help="Lijst van timeframes om te testen")
    parser.add_argument("--optimize", action="store_true",
                        help="Voer alleen optimalisatie uit")
    parser.add_argument("--backtest", action="store_true",
                        help="Voer alleen backtest uit (vereist --params_file of losse parameters)")
    parser.add_argument("--walk_forward", action="store_true",
                        help="Voer alleen walk-forward test uit (vereist --params_file of losse parameters)")
    parser.add_argument("--multi_instrument", action="store_true",
                        help="Voer alleen multi-instrument test uit (vereist --params_file of losse parameters)")
    parser.add_argument("--monte_carlo", action="store_true",
                        help="Voer Monte Carlo analyse uit")
    parser.add_argument("--full_analysis", action="store_true",
                        help="Voer complete analyse uit (default als geen andere actie geselecteerd)")

    # Parameter opties
    parser.add_argument("--params_file", type=str,
                        help="JSON file met parameters voor backtest/walk-forward/multi-instrument")
    parser.add_argument("--window", type=int,
                        help="Window parameter (indien geen params_file)")
    parser.add_argument("--std_dev", type=float,
                        help="Std Dev parameter (indien geen params_file)")
    parser.add_argument("--sl_fixed_percent", type=float,
                        help="Stop-loss percentage (indien geen params_file)")
    parser.add_argument("--tp_fixed_percent", type=float,
                        help="Take-profit percentage (indien geen params_file)")

    args = parser.parse_args()

    # Check of MT5 geïnitialiseerd is
    if not mt5.initialize():
        logger.error("MT5 initialisatie mislukt")
        return

    # Timeframes vertalen naar MT5 constantes
    tf_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1}
    timeframes = {tf: tf_map.get(tf, mt5.TIMEFRAME_D1) for tf in args.timeframes}

    # Parameters ophalen indien nodig
    parameters = {}
    if args.params_file:
        with open(args.params_file, 'r') as f:
            data = json.load(f)
            if 'top_results' in data and len(data['top_results']) > 0:
                parameters = data['top_results'][0]['params']
            else:
                parameters = data['parameters']
    else:
        if args.window:
            parameters['window'] = args.window
        if args.std_dev:
            parameters['std_dev'] = args.std_dev
        if args.sl_fixed_percent:
            parameters['sl_method'] = 'fixed_percent'
            parameters['sl_fixed_percent'] = args.sl_fixed_percent
        if args.tp_fixed_percent:
            parameters['tp_method'] = 'fixed_percent'
            parameters['tp_fixed_percent'] = args.tp_fixed_percent

    # Bepaal welke actie(s) uit te voeren
    if args.optimize:
        logger.info(f"Optimalisatie voor {args.strategy} op {args.symbols[0]}...")

        # Voor elk symbool en timeframe
        for symbol in args.symbols:
            for tf_name, tf_value in timeframes.items():
                logger.info(f"Optimaliseren voor {symbol} ({tf_name})...")
                results = optimize_strategy(strategy_name=args.strategy, symbol=symbol,
                    timeframe=tf_value, metric="sharpe_ratio", top_n=3,
                    ftmo_compliant_only=True)

    elif args.backtest:
        if not parameters:
            logger.error(
                "Backtest vereist parameters via --params_file of losse parameters")
            return

        logger.info(f"Backtest voor {args.strategy} op {args.symbols[0]}...")

        # Voor elk symbool en timeframe
        for symbol in args.symbols:
            for tf_name, tf_value in timeframes.items():
                logger.info(f"Backtest voor {symbol} ({tf_name})...")
                pf, metrics = run_extended_backtest(strategy_name=args.strategy,
                    parameters=parameters, symbol=symbol, timeframe=tf_value)

    elif args.walk_forward:
        if not parameters:
            logger.error(
                "Walk-forward test vereist parameters via --params_file of losse parameters")
            return

        logger.info(f"Walk-forward test voor {args.strategy} op {args.symbols[0]}...")

        # Voor elk symbool
        for symbol in args.symbols:
            logger.info(f"Walk-forward test voor {symbol}...")
            walk_forward_test(strategy_name=args.strategy, symbol=symbol,
                params=parameters, period_days=1095)

    elif args.multi_instrument:
        if not parameters:
            logger.error(
                "Multi-instrument test vereist parameters via --params_file of losse parameters")
            return

        for tf_name, tf_value in timeframes.items():
            logger.info(
                f"Multi-instrument test voor {args.strategy} op timeframe {tf_name}...")
            multi_instrument_test(strategy_name=args.strategy, parameters=parameters,
                instruments=args.symbols, timeframe=tf_value)

    else:
        # Default: volledige analyse
        logger.info(f"Volledige analyse voor {args.strategy}...")
        run_full_analysis(strategy=args.strategy, symbols=args.symbols,
            timeframes=timeframes, monte_carlo=args.monte_carlo)

    # Afsluiten
    mt5.shutdown()
    logger.info("Analyse voltooid!")


if __name__ == "__main__":
    main()