import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import calendar
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from matplotlib.colors import LinearSegmentedColormap

from utils.data_utils import fetch_historical_data
from strategies import get_strategy
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from config import INITIAL_CAPITAL, FEES, OUTPUT_DIR, logger


def calculate_metrics(pf: vbt.Portfolio) -> Dict[str, float]:
    """Bereken portfolio metrics."""
    trades = pf.trades
    has_trades = len(trades) > 0
    return {'total_return': float(pf.total_return()),
        'sharpe_ratio': float(pf.sharpe_ratio()),
        'sortino_ratio': float(pf.sortino_ratio()),
        'calmar_ratio': float(pf.calmar_ratio()),
        'max_drawdown': float(pf.max_drawdown()),
        'win_rate': float(trades.win_rate()) if has_trades else 0.0,
        'trades_count': len(trades), 'profit_factor': float(trades['pnl'].sum() / abs(
            trades['pnl'][trades['pnl'] < 0].sum())) if has_trades and len(
            trades[trades['pnl'] < 0]) > 0 else float('inf'), 'avg_win': float(
            trades.loc[trades['pnl'] > 0, 'pnl'].mean()) if has_trades and len(
            trades[trades['pnl'] > 0]) > 0 else 0.0, 'avg_loss': float(
            trades.loc[trades['pnl'] < 0, 'pnl'].mean()) if has_trades and len(
            trades[trades['pnl'] < 0]) > 0 else 0.0,
        'max_win': float(trades['pnl'].max()) if has_trades else 0.0,
        'max_loss': float(trades['pnl'].min()) if has_trades else 0.0,
        'avg_duration': float(trades['duration'].mean()) if has_trades else 0.0, }


def calculate_income_metrics(pf: vbt.Portfolio, metrics: Dict[str, float],
                             initial_capital: float) -> Dict[str, float]:
    """Bereken inkomstenmetrics op basis van portfolio en initial kapitaal."""
    monthly_returns = pf.returns().resample('M').sum()
    num_years = len(monthly_returns) / 12
    avg_monthly_return = metrics['total_return'] / num_years if num_years > 0 else 0.0
    monthly_income_10k = 10000 * avg_monthly_return
    return {'monthly_returns': monthly_returns.to_dict(),
        'avg_monthly_return': avg_monthly_return,
        'monthly_income_10k': monthly_income_10k,
        'annual_income_10k': monthly_income_10k * 12, }


def create_visualizations(pf: vbt.Portfolio, strategy_name: str, symbol: str,
                          timeframe: Optional[str], output_path: Path,
                          timestamp: str) -> None:
    """Genereer en sla visualisaties op."""
    timeframe_str = f"_{timeframe}" if timeframe else ""

    # Equity curve
    plt.figure(figsize=(12, 6))
    pf.plot().plot()
    plt.title(f"{strategy_name} Equity Curve - {symbol}")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        output_path / f"{strategy_name}_{symbol}{timeframe_str}_equity_{timestamp}.png")
    plt.close()

    # Monthly returns heatmap
    monthly_returns = pf.returns().resample('M').sum()
    years = monthly_returns.index.year.unique()
    heatmap_data = np.full((len(years), 12), np.nan)
    for i, year in enumerate(sorted(years)):
        for j, month in enumerate(range(1, 13)):
            date = pd.Timestamp(year=year, month=month, day=1)
            if date in monthly_returns.index:
                heatmap_data[i, j] = monthly_returns[date]

    plt.figure(figsize=(14, 8))
    cmap = LinearSegmentedColormap.from_list('rg', ["red", "white", "green"], N=256)
    abs_max = np.nanmax(np.abs(heatmap_data))
    im = plt.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-abs_max, vmax=abs_max)
    plt.colorbar(im, format='%.1f%%').set_label('Monthly Return (%)')
    plt.xticks(np.arange(12), [calendar.month_abbr[m] for m in range(1, 13)])
    plt.yticks(np.arange(len(years)), sorted(years))
    plt.title(f"{strategy_name} Monthly Returns - {symbol}")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.savefig(
        output_path / f"{strategy_name}_{symbol}{timeframe_str}_monthly_returns_{timestamp}.png")
    plt.close()


def run_extended_backtest(strategy_name: str, parameters: Dict[str, float], symbol: str,
        timeframe: Optional[str] = None, period_days: int = 1095,
        initial_capital: float = INITIAL_CAPITAL) -> Tuple[
    Optional[vbt.Portfolio], Dict[str, float]]:
    """
    Voer een uitgebreide backtest uit met metrics en visualisaties.

    Args:
        strategy_name: Naam van de strategie.
        parameters: Strategieparameters.
        symbol: Ticker van het instrument.
        timeframe: Timeframe (bijv. 'D1'), optioneel.
        period_days: Aantal dagen historische data.
        initial_capital: Startkapitaal.

    Returns:
        Tuple van (Portfolio-object of None, dictionary met metrics).
    """
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d")

    try:
        # Data ophalen
        df = fetch_historical_data(symbol, timeframe=timeframe, days=period_days)
        if df is None or df.empty:
            logger.error(f"Geen geldige data voor {symbol}")
            return None, {}

        # Strategie en signalen
        strategy = get_strategy(strategy_name, **parameters)
        entries, sl_stop, tp_stop = strategy.generate_signals(df)
        if not all(len(x) == len(df) for x in [entries, sl_stop, tp_stop]):
            raise ValueError("Signalen hebben inconsistente lengtes")

        # Position sizing
        size = None
        if 'risk_per_trade' in parameters:
            size = parameters['risk_per_trade'] / sl_stop.replace(0, np.inf).clip(
                lower=0.01)

        # Portfolio simulatie
        portfolio_kwargs = {'close': df['close'], 'entries': entries,
            'sl_stop': sl_stop, 'tp_stop': tp_stop, 'init_cash': initial_capital,
            'fees': FEES, 'freq': '1D', 'size': size if size is not None else None, }
        if parameters.get('use_trailing_stop',
                          False) and 'trailing_stop_percent' in parameters:
            portfolio_kwargs['sl_trail_stop'] = parameters['trailing_stop_percent']

        pf = vbt.Portfolio.from_signals(**portfolio_kwargs)

        # Metrics berekenen
        metrics = calculate_metrics(pf)
        compliant, profit_target = check_ftmo_compliance(pf, metrics)
        metrics.update(
            {'ftmo_compliant': compliant, 'profit_target_reached': profit_target})
        metrics.update(calculate_income_metrics(pf, metrics, initial_capital))

        # Logging
        logger.info(
            f"\n===== BACKTEST RESULTATEN VOOR {strategy_name} OP {symbol} =====")
        logger.info(f"Totaal rendement: {metrics['total_return']:.2%}")
        logger.info(
            f"Sharpe: {metrics['sharpe_ratio']:.2f}, Max drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(
            f"Win rate: {metrics['win_rate']:.2%}, Trades: {metrics['trades_count']}")

        # Visualisaties
        create_visualizations(pf, strategy_name, symbol, timeframe, output_path,
                              timestamp)

        # Trades en resultaten opslaan
        if len(pf.trades) > 0:
            timeframe_str = f"_{timeframe}" if timeframe else ""
            pf.trades.records_readable.to_csv(
                output_path / f"{strategy_name}_{symbol}{timeframe_str}_trades_{timestamp}.csv")
        with open(
                output_path / f"{strategy_name}_{symbol}{timeframe_str}_results_{timestamp}.json",
                'w') as f:
            json.dump({'strategy': strategy_name, 'symbol': symbol, 'metrics': metrics},
                      f, indent=2)

        return pf, metrics

    except Exception as e:
        logger.error(f"Backtest mislukt voor {symbol}: {str(e)}")
        return None, {}