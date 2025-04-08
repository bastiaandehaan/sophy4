# backtest/extended_backtest.py
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

# Voeg projectroot toe aan pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import fetch_historical_data
from strategies import get_strategy
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from config import INITIAL_CAPITAL, FEES, OUTPUT_DIR, logger


def run_extended_backtest(strategy_name: str, parameters: Dict, symbol: str,
                         timeframe: Optional[str] = None, period_days: int = 1095) -> Tuple[Optional[vbt.Portfolio], Dict]:
    """
    Voer een uitgebreide backtest uit met gedetailleerde metrics en visualisaties.

    Args:
        strategy_name: Naam van de strategie
        parameters: Dictionary met strategieparameters
        symbol: Ticker van het instrument
        timeframe: Timeframe (bijv. 'D1'), optioneel
        period_days: Aantal dagen historische data

    Returns:
        Tuple van (Portfolio-object of None, dictionary met metrics)
    """
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)

    try:
        # Data ophalen
        df = fetch_historical_data(symbol, timeframe=timeframe, days=period_days)
        if df is None or df.empty:
            logger.error(f"Geen geldige data voor {symbol}")
            return None, {}

        # Maak strategie
        strategy = get_strategy(strategy_name, **parameters)

        # Genereer signalen
        entries, sl_stop, tp_stop = strategy.generate_signals(df)
        if not all(len(x) == len(df) for x in [entries, sl_stop, tp_stop]):
            raise ValueError("Signalen hebben inconsistente lengtes")

        # Position sizing op basis van risk_per_trade
        size = None
        if 'risk_per_trade' in parameters:
            size = parameters['risk_per_trade'] / sl_stop.replace(0, np.inf).clip(lower=0.01)
            max_positions = parameters.get('max_positions', float('inf'))
            if entries.sum() > max_positions:
                logger.warning(f"Aantal signalen ({entries.sum()}) overschrijdt max_positions ({max_positions})")

        # Run backtest
        portfolio_kwargs = {
            'close': df['close'],
            'entries': entries,
            'sl_stop': sl_stop,
            'tp_stop': tp_stop,
            'init_cash': INITIAL_CAPITAL,
            'fees': FEES,
            'freq': '1D'
        }

        # Indien size parameter gebruikt wordt:
        if size is not None:
            portfolio_kwargs['size'] = size

        # NIET direct toevoegen: sl_trail=...
        pf = vbt.Portfolio.from_signals(**portfolio_kwargs)

        # Bereken metrics
        metrics = {
            'total_return': float(pf.total_return()),
            'sharpe_ratio': float(pf.sharpe_ratio()),
            'sortino_ratio': float(pf.sortino_ratio()),
            'calmar_ratio': float(pf.calmar_ratio()),
            'max_drawdown': float(pf.max_drawdown()),
            'win_rate': float(pf.trades.win_rate() if len(pf.trades) > 0 else 0),
            'trades_count': len(pf.trades),
            'profit_factor': float(pf.trades['pnl'].sum() / abs(pf.trades['pnl'][pf.trades['pnl'] < 0].sum()))
                           if len(pf.trades[pf.trades['pnl'] < 0]) > 0 else float('inf'),
            'avg_win': float(pf.trades.loc[pf.trades['pnl'] > 0, 'pnl'].mean())
                      if len(pf.trades[pf.trades['pnl'] > 0]) > 0 else 0,
            'avg_loss': float(pf.trades.loc[pf.trades['pnl'] < 0, 'pnl'].mean())
                       if len(pf.trades[pf.trades['pnl'] < 0]) > 0 else 0,
            'max_win': float(pf.trades['pnl'].max()) if len(pf.trades) > 0 else 0,
            'max_loss': float(pf.trades['pnl'].min()) if len(pf.trades) > 0 else 0,
            'avg_duration': float(pf.trades['duration'].mean()) if len(pf.trades) > 0 else 0,
        }

        # FTMO check
        compliant, profit_target = check_ftmo_compliance(pf, metrics)
        metrics['ftmo_compliant'] = compliant
        metrics['profit_target_reached'] = profit_target

        # Logging
        logger.info(f"\n===== BACKTEST RESULTATEN VOOR {strategy_name} OP {symbol} =====")
        logger.info(f"Totaal rendement: {metrics['total_return']:.2%}")
        logger.info(f"Sharpe: {metrics['sharpe_ratio']:.2f}, Sortino: {metrics['sortino_ratio']:.2f}, "
                    f"Calmar: {metrics['calmar_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Win rate: {metrics['win_rate']:.2%}, Trades: {metrics['trades_count']}")
        logger.info(f"FTMO compliant: {'JA' if compliant else 'NEE'}")

        # Visualisaties
        timeframe_str = f"_{timeframe}" if timeframe else ""
        for plot_func, title, filename in [
            (pf.plot, "Equity Curve", "equity"),
            (pf.drawdown, "Drawdowns", "drawdowns")
        ]:
            plt.figure(figsize=(12, 6))
            plot_func().plot()
            plt.title(f"{strategy_name} {title} - {symbol}")
            plt.savefig(output_path / f"{strategy_name}_{symbol}{timeframe_str}_{filename}.png")
            plt.close()

        # Maandelijkse returns heatmap
        try:
            returns_monthly = pf.returns().resample('M').sum()
            plt.figure(figsize=(12, 8))
            plt.imshow([returns_monthly.values], cmap='RdYlGn', aspect='auto')
            plt.colorbar(label='Returns')
            plt.xticks(range(len(returns_monthly)), returns_monthly.index.strftime('%Y-%m'), rotation=90)
            plt.yticks([])
            plt.title(f"{strategy_name} Monthly Returns - {symbol}")
            plt.savefig(output_path / f"{strategy_name}_{symbol}{timeframe_str}_monthly.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Kon geen maandelijkse heatmap maken: {str(e)}")

        # Trade distribution
        if len(pf.trades) > 0:
            plt.figure(figsize=(12, 6))
            pf.trades['pnl'].plot.hist(bins=20, alpha=0.7)
            plt.axvline(0, color='r', linestyle='--')
            plt.title(f"{strategy_name} Trade Distribution - {symbol}")
            plt.savefig(output_path / f"{strategy_name}_{symbol}{timeframe_str}_trades_dist.png")
            plt.close()

        # Sla trades en resultaten op
        if len(pf.trades) > 0:
            pf.trades.records_readable.to_csv(output_path / f"{strategy_name}_{symbol}{timeframe_str}_trades.csv")
        with open(output_path / f"{strategy_name}_{symbol}{timeframe_str}_results.json", 'w') as f:
            json.dump({'strategy': strategy_name, 'symbol': symbol, 'timeframe': str(timeframe),
                      'parameters': parameters, 'metrics': metrics}, f, indent=2)

        logger.info(f"Resultaten en grafieken opgeslagen in {output_path}")
        return pf, metrics

    except Exception as e:
        logger.error(f"Backtest mislukt voor {symbol}: {str(e)}")
        return None, {}