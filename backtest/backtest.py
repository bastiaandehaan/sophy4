# backtest/backtest.py - Updated to handle Kelly sizing
import logging
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from strategies import get_strategy
from config import logger, config_manager
from utils.data_utils import fetch_historical_data


def run_extended_backtest(
    strategy_name: str,
    parameters: Dict[str, Any],
    symbol: str,
    timeframe: str,
    period_days: int,
    initial_capital: float = 10000.0,
    output_path: Path = None,
    silent: bool = False,
    timestamp: str = None,
    end_date: Optional[datetime] = None,
    original_level: int = 0,
) -> Tuple[vbt.Portfolio, Dict[str, Any]]:
    """
    Run an extended backtest for a given strategy with historical data.
    Updated to handle dynamic position sizing (e.g., Kelly sizing).
    """
    # Fetch historical data
    if end_date is None:
        end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=period_days)

    logger.info(
        f"Starting backtest: {strategy_name} on {symbol} with {period_days} days data"
    )
    logger.info(
        f"Fetching data for {symbol}, timeframe: {timeframe}, from {start_date} to {end_date}"
    )

    df = fetch_historical_data(symbol, timeframe, start_date, end_date)

    if df is None or df.empty:
        logger.error(f"No data available for {symbol} on timeframe {timeframe}")
        raise ValueError(f"No data available for {symbol} on timeframe {timeframe}")

    logger.info(f"Data range: from {df.index[0]} to {df.index[-1]}")
    logger.info(
        f"Historical data loaded: {len(df)} rows, columns: {list(df.columns)}"
    )

    # Generate signals
    strategy = get_strategy(strategy_name, **parameters)
    # Updated to handle four return values including sizes
    entries, sl_stop, tp_stop, sizes = strategy.generate_signals(df)

    # Run vectorbt backtest with dynamic sizing
    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries.astype(bool),
        size=sizes,  # Use the dynamic sizes from Kelly criterion
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        freq=timeframe,
        init_cash=initial_capital,
        fees=parameters.get("fees", 0.0001),  # 0.01% fees
        slippage=parameters.get("slippage", 0.0005),  # 0.05% slippage
    )

    # Calculate metrics
    total_return = pf.total_return()
    sharpe_ratio = pf.sharpe_ratio()
    sortino_ratio = pf.sortino_ratio()
    max_drawdown = pf.max_drawdown()
    cagr = pf.annualized_return()
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.inf

    trades = pf.trades.records_readable
    win_rate = pf.win_rate()
    trades_count = len(trades)
    avg_winning_trade = (
        trades[trades["PnL"] > 0]["PnL"].mean() if trades["PnL"].gt(0).any() else 0
    )
    avg_losing_trade = (
        trades[trades["PnL"] <= 0]["PnL"].mean() if trades["PnL"].le(0).any() else 0
    )
    profit_factor = pf.profit_factor()
    absolute_profit = pf.total_profit()

    avg_profit_per_trade = absolute_profit / trades_count if trades_count > 0 else 0
    duration = (df.index[-1] - df.index[0]).days / 30  # Approximate months
    avg_monthly_profit = absolute_profit / duration if duration > 0 else 0

    # Check FTMO compliance (simplified)
    ftmo_compliant = max_drawdown > -0.10  # Max drawdown < 10%
    profit_target_reached = total_return >= 0.10  # 10% profit target

    metrics = {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "cagr": cagr,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "trades_count": trades_count,
        "avg_winning_trade": avg_winning_trade,
        "avg_losing_trade": avg_losing_trade,
        "profit_factor": profit_factor,
        "absolute_profit": absolute_profit,
        "avg_profit_per_trade": avg_profit_per_trade,
        "avg_monthly_profit": avg_monthly_profit,
        "ftmo_compliant": ftmo_compliant,
        "profit_target_reached": profit_target_reached,
    }

    # Save results
    if output_path:
        timestamp = timestamp or datetime.now().strftime("%Y%m%d")
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_file = output_path / f"{strategy_name}_{symbol}_{timeframe}_results_{timestamp}.json"
        trades_file = output_path / f"{strategy_name}_{symbol}_{timeframe}_trades_{timestamp}.csv"

        # Save metrics
        pd.Series(metrics).to_json(metrics_file, indent=4)

        # Save trades
        if not trades.empty:
            trades.to_csv(trades_file, index=False)

        if not silent:
            logger.info(f"Backtest results saved to {metrics_file}")
            logger.info(f"Trade details saved to {trades_file}")

    return pf, metrics