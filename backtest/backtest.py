import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import vectorbt as vbt

from backtest.data_loader import fetch_historical_data
from config import INITIAL_CAPITAL, FEES, OUTPUT_DIR, logger
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from strategies import get_strategy
from utils.plotting import create_visualizations


def _calculate_stop(portfolio_kwargs: Dict[str, Any],
                    parameters: Dict[str, Any]) -> None:
    """Calculate stop losses and trailing stops."""
    if parameters.get('use_trailing_stop', False):
        trailing_stop = parameters.get('trailing_stop_percent', 0.02)
        portfolio_kwargs['sl_stop'] = portfolio_kwargs['sl_stop'].fillna(trailing_stop)
        portfolio_kwargs['sl_trail'] = True
        trail_start = parameters.get('trailing_activation_percent', 0)
        if trail_start > 0:
            portfolio_kwargs['sl_trail_start'] = trail_start
    else:
        portfolio_kwargs['sl_trail'] = False


def calculate_metrics(pf: vbt.Portfolio) -> Dict[str, Any]:
    """Calculate performance metrics for a portfolio."""
    metrics = {}
    metrics['total_return'] = float(pf.total_return())
    metrics['sharpe_ratio'] = float(pf.sharpe_ratio()) if not np.isnan(
        pf.sharpe_ratio()) else 0.0
    metrics['sortino_ratio'] = float(pf.sortino_ratio()) if not np.isnan(
        pf.sortino_ratio()) else 0.0
    metrics['max_drawdown'] = float(pf.max_drawdown()) if not np.isnan(
        pf.max_drawdown()) else 0.0
    metrics['cagr'] = float(pf.annualized_return()) if not np.isnan(
        pf.annualized_return()) else 0.0
    metrics['calmar_ratio'] = abs(metrics['cagr'] / metrics['max_drawdown']) if metrics[
                                                                                    'cagr'] > 0 and \
                                                                                metrics[
                                                                                    'max_drawdown'] < 0 else 0.0

    if len(pf.trades) > 0:
        metrics['win_rate'] = float(pf.trades.win_rate())
        metrics['trades_count'] = len(pf.trades)
        metrics['avg_winning_trade'] = float(pf.trades.winning.pnl.mean()) if len(
            pf.trades.winning) > 0 else 0.0
        metrics['avg_losing_trade'] = float(pf.trades.losing.pnl.mean()) if len(
            pf.trades.losing) > 0 else 0.0
        total_win = float(pf.trades.winning.pnl.sum()) if len(
            pf.trades.winning) > 0 else 0.0
        total_loss = float(abs(pf.trades.losing.pnl.sum())) if len(
            pf.trades.losing) > 0 else 0.0
        metrics['profit_factor'] = total_win / total_loss if total_loss > 0 else float(
            'inf')
    else:
        metrics['win_rate'] = 0.0
        metrics['trades_count'] = 0
        metrics['avg_winning_trade'] = 0.0
        metrics['avg_losing_trade'] = 0.0
        metrics['profit_factor'] = 0.0

    return metrics


def calculate_income_metrics(pf: vbt.Portfolio, metrics: Dict[str, Any],
                             initial_capital: float) -> Dict[str, Any]:
    """Calculate absolute income metrics."""
    income_metrics = {}
    income_metrics['absolute_profit'] = float(initial_capital * metrics['total_return'])
    income_metrics['avg_profit_per_trade'] = float(
        income_metrics['absolute_profit'] / metrics['trades_count']) if metrics[
                                                                            'trades_count'] > 0 else 0.0
    days = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days
    months = days / 30.0 if days > 0 else 1.0
    income_metrics['avg_monthly_profit'] = float(
        income_metrics['absolute_profit'] / months)
    return income_metrics


def run_extended_backtest(strategy_name: str, parameters: Dict[str, Any], symbol: str,
                          timeframe: Optional[Union[str, int]] = None,
                          period_days: int = 1095,
                          initial_capital: float = INITIAL_CAPITAL,
                          end_date: Optional[datetime] = None, silent: bool = False) -> \
Tuple[Optional[vbt.Portfolio], Dict[str, Any]]:
    """Perform an extended backtest with VectorBT."""

    if not silent:
        print(f"\n{'=' * 60}")
        print(f"=== START BACKTEST: {strategy_name} on {symbol} ===")
        print(f"{'=' * 60}")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d")

    # Handle logging
    original_level = logger.level
    if silent:
        import logging
        logger.setLevel(logging.ERROR)
    else:
        logger.info(
            f"Starting backtest: {strategy_name} on {symbol} with {period_days} days data")

    # Fetch data
    df = fetch_historical_data(symbol, timeframe=timeframe, days=period_days,
                               end_date=end_date)
    if df is None or df.empty:
        logger.error(f"No valid data for {symbol}")
        return None, {}

    # Generate signals
    strategy = get_strategy(strategy_name, **parameters)
    entries, sl_stop, tp_stop = strategy.generate_signals(df)

    # Simple position size calculation
    risk_per_trade = parameters.get('risk_per_trade', 0.01)
    risk_amount = initial_capital * risk_per_trade
    current_price = df['close'].iloc[-1]
    assumed_sl_pct = 0.02
    price_risk = current_price * assumed_sl_pct
    pip_value = 10.0

    size = risk_amount / (price_risk * pip_value) if price_risk > 0 else 0.01
    size = max(0.01, min(size, 10.0))

    # Map timeframe to freq
    timeframe_to_freq = {'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
                         'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w', 'MN1': '1M'}
    freq = timeframe_to_freq.get(str(timeframe), '1d')

    # Create portfolio with VectorBT
    portfolio_kwargs = {'close': df['close'], 'entries': entries > 0,
        'short_entries': entries < 0, 'sl_stop': sl_stop, 'tp_stop': tp_stop,
        'init_cash': initial_capital, 'fees': FEES, 'freq': freq, 'size': size,
        'size_type': 'amount'}
    _calculate_stop(portfolio_kwargs, parameters)

    try:
        pf = vbt.Portfolio.from_signals(**portfolio_kwargs)
    except Exception as e:
        logger.error(f"Portfolio creation failed: {str(e)}")
        return None, {}

    # Calculate metrics
    metrics = calculate_metrics(pf)
    income_metrics = calculate_income_metrics(pf, metrics, initial_capital)
    compliant, profit_target = check_ftmo_compliance(pf, metrics)
    all_metrics = {**metrics, **income_metrics, 'ftmo_compliant': compliant,
                   'profit_target_reached': profit_target}

    # Log results
    if not silent:
        logger.info(f"\n===== BACKTEST RESULTS FOR {strategy_name} ON {symbol} =====")
        logger.info(f"Total Return: {float(metrics['total_return']):.2%}")
        logger.info(
            f"Sharpe: {float(metrics['sharpe_ratio']):.2f}, Max Drawdown: {float(metrics['max_drawdown']):.2%}")
        logger.info(
            f"Win Rate: {float(metrics['win_rate']):.2%}, Trades: {metrics['trades_count']}")
        logger.info(
            f"FTMO Compliant: {'YES' if compliant else 'NO'}, Profit Target Reached: {'YES' if profit_target else 'NO'}")

        # Create visualizations
        try:
            pf.plot().show()
            create_visualizations(pf, strategy_name, symbol, timeframe, output_path,
                                  timestamp)
        except Exception as e:
            logger.warning(f"Visualizations could not be created: {str(e)}")

        # Save results
        timeframe_str = f"_{timeframe}" if timeframe else ""
        if len(pf.trades) > 0:
            pf.trades.records_readable.to_csv(
                output_path / f"{strategy_name}_{symbol}{timeframe_str}_trades_{timestamp}.csv")

        with open(
                output_path / f"{strategy_name}_{symbol}{timeframe_str}_results_{timestamp}.json",
                'w') as f:
            json.dump({'strategy': strategy_name, 'symbol': symbol,
                       'timeframe': str(timeframe), 'metrics': all_metrics}, f,
                      indent=2)

        # Print results
        print("\n=== BACKTEST RESULTS ===")
        print(f"Total Return: {float(metrics['total_return']):.2%}")
        print(f"Sharpe Ratio: {float(metrics['sharpe_ratio']):.2f}")
        print(f"Max Drawdown: {float(metrics['max_drawdown']):.2%}")
        print(f"Win Rate: {float(metrics['win_rate']):.2%}")
        print(f"Number of Trades: {metrics['trades_count']}")
        print(f"FTMO Compliant: {'YES' if compliant else 'NO'}")

    # Restore logger level
    if silent:
        logger.setLevel(original_level)

    return pf, all_metrics