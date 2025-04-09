#!/usr/bin/env python
"""
Script om een backtest uit te voeren met parameters uit een JSON bestand.
Gebruik: python run_backtest_from_json.py --json_file results/BollongStrategy_GER40.cash_quick_optim.json [--index 0]
"""
import argparse
import json
import sys
from pathlib import Path

# Voeg projectroot toe aan Python path
sys.path.append(str(Path(__file__).parent))

from backtest.extended_backtest import run_extended_backtest
from config import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Uitgebreide backtest met geoptimaliseerde parameters")
    parser.add_argument("--json_file", type=str, required=True,
                        help="JSON bestand met geoptimaliseerde parameters")
    parser.add_argument("--index", type=int, default=0,
                        help="Index van de parameterset in het JSON bestand (default: 0 = beste set)")
    parser.add_argument("--symbol", type=str, default="GER40.cash",
                        help="Handelssymbool (default: GER40.cash)")
    parser.add_argument("--timeframe", type=str, default="D1",
                        help="Timeframe (default: D1)")
    parser.add_argument("--period_days", type=int, default=1095,
                        help="Aantal dagen historische data (default: 1095)")
    parser.add_argument("--initial_capital", type=float, default=10000.0,
                        help="Initieel kapitaal (default: 10000.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Laad parameters uit JSON
    try:
        with open(args.json_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Kon JSON bestand niet laden: {e}")
        return

    if not results or args.index >= len(results):
        logger.error(f"Geen resultaten gevonden of index {args.index} is buiten bereik")
        return

    # Haal parameterset op
    param_set = results[args.index]
    params = param_set['params']
    metrics = param_set['metrics']

    logger.info("\n" + "=" * 80)
    logger.info(f"BACKTEST MET GEOPTIMALISEERDE PARAMETERS (index {args.index}):")
    logger.info(
        f"Symbool: {args.symbol}, Timeframe: {args.timeframe}, Periode: {args.period_days} dagen")
    logger.info("-" * 80)
    logger.info("Parameters:")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")
    logger.info("-" * 80)
    logger.info("Verwachte metrics op basis van optimalisatie:")
    for k, v in metrics.items():
        if k in ['total_return', 'max_drawdown', 'win_rate']:
            logger.info(f"  {k}: {v * 100:.2f}%")
        elif k in ['sharpe_ratio', 'calmar_ratio', 'sortino_ratio']:
            logger.info(f"  {k}: {v:.2f}")
        elif k in ['trades_count', 'signal_count', 'trade_count']:
            logger.info(f"  {k}: {int(v)}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info("=" * 80 + "\n")

    # Voer backtest uit
    logger.info("Backtest wordt uitgevoerd, even geduld...")

    # Bepaal strategie naam uit het bestandsnaam
    strategy_name = Path(args.json_file).stem.split('_')[0]

    # Voer backtest uit
    pf, backtest_metrics = run_extended_backtest(strategy_name=strategy_name,
        parameters=params, symbol=args.symbol, timeframe=args.timeframe,
        period_days=args.period_days, initial_capital=args.initial_capital)

    if pf is None or not backtest_metrics:
        logger.error("Backtest mislukt. Controleer logs voor details.")
        return

    # Toon resultaten
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTATEN:")
    logger.info("-" * 80)

    # Vergelijk de resultaten met de verwachte metrics
    logger.info("Metric              | Verwacht    | Behaald      | Verschil")
    logger.info("-" * 80)

    # Belangrijkste metrics weergeven met vergelijking
    for metric_name, expected_label, format_str in [
        ('total_return', 'Return', '{:.2%}'), ('sharpe_ratio', 'Sharpe', '{:.2f}'),
        ('calmar_ratio', 'Calmar', '{:.2f}'), ('sortino_ratio', 'Sortino', '{:.2f}'),
        ('max_drawdown', 'Max DD', '{:.2%}'), ('win_rate', 'Win Rate', '{:.2%}'),
        ('trades_count', 'Trades', '{:.0f}'), ]:
        expected = metrics.get(metric_name, 0)
        actual = backtest_metrics.get(metric_name, 0)

        # Voor percentages
        if metric_name in ['total_return', 'max_drawdown', 'win_rate']:
            expected_fmt = format_str.format(expected)
            actual_fmt = format_str.format(actual)
            diff = (actual - expected) * 100
            diff_fmt = f"{diff:+.2f}%"
        else:
            expected_fmt = format_str.format(expected)
            actual_fmt = format_str.format(actual)
            diff = actual - expected
            diff_fmt = f"{diff:+.2f}"

        logger.info(
            f"{expected_label:18} | {expected_fmt:12} | {actual_fmt:12} | {diff_fmt}")

    logger.info("=" * 80)
    logger.info(
        "Backtest voltooid. Alle resultaten zijn opgeslagen in de results directory.")


if __name__ == "__main__":
    main()