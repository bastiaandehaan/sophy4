#!/usr/bin/env python
"""
Sophy4 Trading Framework - CLI Application
"""
import os
# Onderdruk TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path
# Fix import path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
from typing import Optional, List
import json

import typer
from rich.console import Console
from rich.table import Table
import pandas as pd

from backtest.backtest import run_extended_backtest
from config import logger, INITIAL_CAPITAL, SYMBOLS
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from strategies import STRATEGIES, list_strategies

# Initialize Typer app
app = typer.Typer(name="sophy4",
    help="Sophy4 Trading Framework - Professional trading with FTMO compliance",
    add_completion=True)
console = Console()

# Create subcommands
backtest_app = typer.Typer(help="Backtesting operations")
optimize_app = typer.Typer(help="Strategy optimization")
monitor_app = typer.Typer(help="Performance monitoring")
strategy_app = typer.Typer(help="Strategy management")

app.add_typer(backtest_app, name="backtest")
app.add_typer(optimize_app, name="optimize")
app.add_typer(monitor_app, name="monitor")
app.add_typer(strategy_app, name="strategy")


@backtest_app.command("run")
def run_backtest(strategy: str = typer.Argument(..., help="Strategy name to use"),
        symbol: str = typer.Option(SYMBOLS[0], "--symbol", "-s", help="Trading symbol"),
        timeframe: str = typer.Option("H1", "--timeframe", "-t",
                                      help="Timeframe (M5, H1, D1, etc.)"),
        days: int = typer.Option(1095, "--days", "-d",
                                 help="Number of days of historical data"),
        params_file: Optional[Path] = typer.Option(None, "--params", "-p",
                                                   help="JSON file with parameters"),
        params_index: int = typer.Option(0, "--index", "-i",
                                         help="Index in params file (0=best)"),
        initial_capital: float = typer.Option(INITIAL_CAPITAL, "--capital", "-c",
                                              help="Initial capital")):
    """Run a backtest with specified strategy and parameters."""
    # Validate strategy
    if strategy not in STRATEGIES:
        console.print(f"[red]Error:[/red] Strategy '{strategy}' not found")
        console.print(f"Available strategies: {', '.join(STRATEGIES.keys())}")
        raise typer.Exit(1)

    # Validate symbol
    if symbol not in SYMBOLS:
        console.print(f"[red]Error:[/red] Symbol '{symbol}' not configured")
        console.print(f"Available symbols: {', '.join(SYMBOLS)}")
        raise typer.Exit(1)

    # Load parameters
    parameters = {}
    if params_file and params_file.exists():
        try:
            with open(params_file) as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > params_index:
                    parameters = data[params_index]['params']
                    console.print(
                        f"[green]Loaded parameters from {params_file} (index {params_index})[/green]")
        except Exception as e:
            console.print(f"[red]Error loading parameters:[/red] {e}")
            raise typer.Exit(1)

    # Handle direct parameter input
    if not parameters:
        parameters = {'symbol': symbol}  # Make sure symbol is always passed

    # Run backtest
    with console.status("[bold green]Running backtest...") as status:
        pf, metrics = run_extended_backtest(strategy_name=strategy,
            parameters=parameters, symbol=symbol, timeframe=timeframe, period_days=days,
            initial_capital=initial_capital)

    if pf is None:
        console.print("[red]Backtest failed![/red]")
        raise typer.Exit(1)

    # Display results
    table = Table(title=f"Backtest Results - {strategy} on {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Return", f"{metrics.get('total_return', 0):.2%}")
    table.add_row("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    table.add_row("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    table.add_row("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
    table.add_row("Number of Trades", str(metrics.get('trades_count', 0)))

    # FTMO compliance check
    compliant, profit_target = check_ftmo_compliance(pf, metrics)
    table.add_row("FTMO Compliant", "✅ Yes" if compliant else "❌ No")
    table.add_row("Profit Target Reached", "✅ Yes" if profit_target else "❌ No")

    console.print(table)


@strategy_app.command("list")
def list_available_strategies():
    """List all available trading strategies."""
    strategies = list_strategies()

    table = Table(title="Available Trading Strategies")
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Parameters", style="green")
    table.add_column("Description", style="yellow")

    for strategy in strategies:
        params_str = "\n".join([f"• {p}" for p in strategy['accepted_params']])
        desc_str = "\n".join(
            [f"{k}: {v}" for k, v in strategy['descriptions'].items()][:3])
        table.add_row(strategy['name'], params_str, desc_str)

    console.print(table)


@optimize_app.command("run")
def run_optimization(strategy: str = typer.Argument(..., help="Strategy to optimize"),
        symbol: str = typer.Option(SYMBOLS[0], "--symbol", "-s", help="Trading symbol"),
        timeframe: str = typer.Option("D1", "--timeframe", "-t", help="Timeframe"),
        days: int = typer.Option(365, "--days", "-d", help="Historical data days"),
        metric: str = typer.Option("sharpe_ratio", "--metric", "-m",
                                   help="Optimization metric"),
        quick: bool = typer.Option(False, "--quick", "-q",
                                   help="Quick mode with fewer params"),
        top_n: int = typer.Option(5, "--top", help="Number of top results to show")):
    """Run parameter optimization for a strategy."""
    from optimization.optimize import quick_optimize

    with console.status("[bold green]Running optimization...") as status:
        r