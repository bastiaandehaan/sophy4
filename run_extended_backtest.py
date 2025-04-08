#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple  # Toegevoegd: Tuple voor type hint
import vectorbt as vbt  # Toegevoegd: vectorbt import voor vbt.Portfolio

from backtest.extended_backtest import run_extended_backtest


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy4 Extended Backtesting Tool")
    parser.add_argument("--strategy", type=str, required=True,
                        help="Naam van de strategie")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbool")
    parser.add_argument("--timeframe", type=str, default="D1",
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


def run_backtest(args: argparse.Namespace) -> Tuple[Optional[vbt.Portfolio], Dict[str, float]]:
    """Voer de backtest uit met de opgegeven argumenten."""
    strategy_params: Dict[str, float] = {
        'window': args.window,
        'std_dev': args.std_dev,
        'risk_per_trade': args.risk_per_trade,
    }

    print(f"\nRunning backtest: {args.strategy} on {args.symbol} ({args.timeframe})")
    return run_extended_backtest(
        strategy_name=args.strategy,
        parameters=strategy_params,
        symbol=args.symbol,
        timeframe=args.timeframe,
        period_days=args.period_days,
        initial_capital=args.initial_capital,
    )


def display_results(portfolio: Optional[vbt.Portfolio], metrics: Dict[str, float]) -> None:
    """Toon de backtestresultaten."""
    if portfolio is None or not metrics:
        print("Backtest mislukt. Controleer logs.")
        return

    print("\n=== RESULTATEN ===")
    print(f"Totaal rendement: {metrics['total_return']:.2%}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Aantal trades: {metrics['trades_count']}")
    print(f"Maandelijkse inkomsten (€10k): €{metrics['monthly_income_10k']:.2f}")
    print(f"Jaarlijkse inkomsten (€10k): €{metrics['annual_income_10k']:.2f}")


def main() -> None:
    """Hoofdfunctie voor het uitvoeren van de backtest."""
    args = parse_args()
    portfolio, metrics = run_backtest(args)
    display_results(portfolio, metrics)
    print("\nBacktest voltooid. Resultaten opgeslagen in output directory.")


if __name__ == "__main__":
    main()