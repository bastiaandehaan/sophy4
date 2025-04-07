# backtest/extended_backtest.py
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
from typing import Dict, Tuple, Optional

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

        # Voeg trailing stop toe
        sl_trail = getattr(strategy, 'sl_trail', False)

        # Position sizing op basis van risk_per_trade
        size = None
        if 'risk_per_trade' in parameters:
            size = parameters['risk_per_trade'] / sl_stop.replace(0, np.inf).clip(lower=0.01)
            max_positions = parameters.get('max_positions', float('inf'))
            if entries.sum() > max_positions:
                logger.warning(f"Aantal signalen ({entries.sum()}) overschrijdt max_positions ({max_positions})")

        # Run backtest
        pf = vbt.Portfolio.from_signals(
            close=df['close'],
            entries=entries,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            sl_trail=sl_trail,
            init_cash=INITIAL_CAPITAL,
            fees=FEES,
            freq='1D',
            size=size
        )

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


def monte_carlo_analysis(pf: vbt.Portfolio, n_simulations: int = 1000) -> Optional[Dict]:
    """
    Voer een Monte Carlo-analyse uit om de robuustheid van de strategie te testen.

    Args:
        pf: VectorBT Portfolio-object
        n_simulations: Aantal simulaties

    Returns:
        Dictionary met Monte Carlo-statistieken of None bij te weinig trades
    """
    if len(pf.trades) < 10:
        logger.warning("Te weinig trades voor Monte Carlo-analyse (< 10)")
        return None

    trades = pf.trades.records
    returns = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    sharpe_ratios = np.zeros(n_simulations)

    for i in range(n_simulations):
        shuffled_indices = np.random.permutation(len(trades))
        shuffled_trades = trades.iloc[shuffled_indices].reset_index(drop=True)

        equity = INITIAL_CAPITAL
        equity_curve = [INITIAL_CAPITAL]
        peak = INITIAL_CAPITAL
        drawdowns = []

        for _, trade in shuffled_trades.iterrows():
            equity += trade['pnl']
            equity_curve.append(equity)
            peak = max(peak, equity)
            drawdowns.append((equity - peak) / peak if peak > 0 else 0)

        equity_series = pd.Series(equity_curve, index=pd.date_range(start='2020-01-01', periods=len(equity_curve), freq='D'))
        returns[i] = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
        max_drawdowns[i] = min(drawdowns) if drawdowns else 0
        sharpe_ratios[i] = vbt.Portfolio.from_orders(
            close=equity_series, size=np.diff(equity_curve, prepend=INITIAL_CAPITAL),
            init_cash=INITIAL_CAPITAL, freq='1D'
        ).sharpe_ratio()

    results = {
        'return_mean': float(np.mean(returns)), 'return_median': float(np.median(returns)),
        'return_std': float(np.std(returns)), 'return_min': float(np.min(returns)),
        'return_max': float(np.max(returns)), 'return_5pct': float(np.percentile(returns, 5)),
        'return_95pct': float(np.percentile(returns, 95)),
        'max_drawdown_mean': float(np.mean(max_drawdowns)), 'max_drawdown_median': float(np.median(max_drawdowns)),
        'max_drawdown_std': float(np.std(max_drawdowns)), 'max_drawdown_worst': float(np.min(max_drawdowns)),
        'max_drawdown_5pct': float(np.percentile(max_drawdowns, 5)),
        'max_drawdown_95pct': float(np.percentile(max_drawdowns, 95)),
        'sharpe_mean': float(np.mean(sharpe_ratios)), 'sharpe_median': float(np.median(sharpe_ratios)),
        'sharpe_std': float(np.std(sharpe_ratios)),
    }

    output_path = Path(OUTPUT_DIR)
    for data, title, filename in [
        (returns, "Return Distribution", "monte_carlo_returns"),
        (max_drawdowns, "Max Drawdown Distribution", "monte_carlo_drawdowns")
    ]:
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=50, alpha=0.7)
        plt.axvline(np.percentile(data, 5), color='r', linestyle='--', label='5% Percentile')
        plt.axvline(np.mean(data), color='g', linestyle='-', label='Mean')
        plt.axvline(np.percentile(data, 95), color='r', linestyle='--', label='95% Percentile')
        plt.title(f"Monte Carlo: {title}")
        plt.legend()
        plt.savefig(output_path / f"{filename}.png")
        plt.close()

    logger.info(f"\n===== MONTE CARLO RESULTATEN ({n_simulations} SIMULATIES) =====")
    logger.info(f"Gemiddeld rendement: {results['return_mean']:.2%}")
    logger.info(f"5% - 95% rendement: {results['return_5pct']:.2%} - {results['return_95pct']:.2%}")
    logger.info(f"Gemiddelde max drawdown: {results['max_drawdown_mean']:.2%}")
    logger.info(f"Gemiddelde Sharpe: {results['sharpe_mean']:.2f}")

    return results


def main():
    import argparse
    import MetaTrader5 as mt5

    parser = argparse.ArgumentParser(description="Sophy4 Extended Backtest")
    parser.add_argument("--strategy", type=str, required=True, help="Naam van de strategie")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbool")
    parser.add_argument("--timeframe", type=str, choices=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1'], default='D1')
    parser.add_argument("--params_file", type=str, help="JSON file met parameters (optioneel)")
    parser.add_argument("--window", type=int, help="Window parameter (indien geen params_file)")
    parser.add_argument("--std_dev", type=float, help="Std Dev parameter (indien geen params_file)")
    parser.add_argument("--sl_fixed_percent", type=float, help="Stop-loss percentage (indien geen params_file)")
    parser.add_argument("--tp_fixed_percent", type=float, help="Take-profit percentage (indien geen params_file)")
    parser.add_argument("--monte_carlo", action="store_true", help="Voer Monte Carlo analyse uit")

    args = parser.parse_args()
    tf_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
              'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
              'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1}
    timeframe = tf_map.get(args.timeframe, mt5.TIMEFRAME_D1)

    if args.params_file:
        with open(args.params_file, 'r') as f:
            data = json.load(f)
            parameters = data.get('top_results', [{}])[0].get('params', data.get('parameters', {}))
    else:
        parameters = {
            'window': args.window, 'std_dev': args.std_dev,
            'sl_method': 'fixed_percent' if args.sl_fixed_percent else None,
            'sl_fixed_percent': args.sl_fixed_percent,
            'tp_method': 'fixed_percent' if args.tp_fixed_percent else None,
            'tp_fixed_percent': args.tp_fixed_percent
        }
        parameters = {k: v for k, v in parameters.items() if v is not None}

    pf, metrics = run_extended_backtest(args.strategy, parameters, args.symbol, timeframe)
    if args.monte_carlo and pf is not None:
        monte_carlo_analysis(pf)


if __name__ == "__main__":
    main()