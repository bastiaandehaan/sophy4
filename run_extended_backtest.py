#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import vectorbt as vbt

from backtest.extended_backtest import run_extended_backtest
from config import logger  # Toegevoegd voor consistente logging
from utils.data_utils import TIMEFRAME_MAP  # Toegevoegd voor timeframe-conversie


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy4 Extended Backtesting Tool")
    parser.add_argument("--strategy", type=str, required=True,
                        help="Naam van de strategie")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbool")
    parser.add_argument("--timeframe", type=str, default="D1",
                        choices=list(TIMEFRAME_MAP.keys()),
                        help="Timeframe (bijv. H1, D1)")
    parser.add_argument("--period_days", type=int, default=1095,
                        help="Aantal dagen historische data")
    parser.add_argument("--initial_capital", type=float, default=10000.0,
                        help="Initieel kapitaal")

    # Strategieparameters
    parser.add_argument("--window", type=int, default=20, help="Bollinger Band window")
    parser.add_argument("--std_dev", type=float, default=2.0,
                        help="Bollinger Band standaarddeviaties")
    parser.add_argument("--risk_per_trade", type=float, default=0.01,
                        help="Risico per trade (default: 1%)")

    return parser.parse_args()


def run_backtest(args: argparse.Namespace) -> Tuple[Optional[vbt.Portfolio], Dict[str, Any]]:
    """Voer de backtest uit met de opgegeven argumenten."""
    strategy_params: Dict[str, float] = {
        'window': args.window,
        'std_dev': args.std_dev,
        'risk_per_trade': args.risk_per_trade,
    }

    logger.info(f"Running backtest: {args.strategy} on {args.symbol} ({args.timeframe})")
    return run_extended_backtest(
        strategy_name=args.strategy,
        parameters=strategy_params,
        symbol=args.symbol,
        timeframe=args.timeframe,  # String wordt doorgegeven, conversie in data_utils
        period_days=args.period_days,
        initial_capital=args.initial_capital,
    )


def display_results(portfolio: Optional[vbt.Portfolio], metrics: Dict[str, Any]) -> None:
    """Toon de backtestresultaten."""
    if portfolio is None or not metrics:
        logger.error("Backtest mislukt. Controleer logs voor details.")
        return

    logger.info("\n=== RESULTATEN ===")
    try:
        logger.info(f"Totaal rendement: {float(metrics['total_return']):.2%}")
        logger.info(f"Sharpe ratio: {float(metrics['sharpe_ratio']):.2f}")
        logger.info(f"Max drawdown: {float(metrics['max_drawdown']):.2%}")
        logger.info(f"Win rate: {float(metrics['win_rate']):.2%}")
        logger.info(f"Aantal trades: {int(metrics['trades_count'])}")
        logger.info(f"Maandelijkse inkomsten (€10k): €{float(metrics['monthly_income_10k']):.2f}")
        logger.info(f"Jaarlijkse inkomsten (€10k): €{float(metrics['annual_income_10k']):.2f}")
        logger.info(f"FTMO compliant: {'JA' if metrics['ftmo_compliant'] else 'NEE'}")
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Fout bij weergeven resultaten: {str(e)}. Metrics: {metrics}")


def main() -> None:
    """Hoofdfunctie voor het uitvoeren van de backtest."""
    args = parse_args()
    portfolio, metrics = run_backtest(args)
    display_results(portfolio, metrics)
    logger.info("Backtest voltooid. Resultaten opgeslagen in output directory.")


if __name__ == "__main__":
    main()