# backtest/extended_backtest.py
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt

# Voeg projectroot toe aan pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import fetch_historical_data
from strategies import get_strategy
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from config import INITIAL_CAPITAL, FEES, OUTPUT_DIR, logger


def run_extended_backtest(strategy_name, parameters, symbol, timeframe=None,
                          period_days=1095):
    """
    Voer een uitgebreide backtest uit met veel details en visualisaties.
    """
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)

    # Data ophalen
    df = fetch_historical_data(symbol, timeframe=timeframe, days=period_days)
    if df is None:
        logger.error(f"Geen data voor {symbol}")
        return None

    # Maak strategie
    strategy = get_strategy(strategy_name, **parameters)

    # Genereer signalen
    entries, sl_stop, tp_stop = strategy.generate_signals(df)

    # Voeg trail_stop toe als het beschikbaar is
    kwargs = {}
    if hasattr(strategy, 'trail_stop') and strategy.trail_stop is not None:
        kwargs['trail_stop'] = strategy.trail_stop

    # Run backtest
    pf = vbt.Portfolio.from_signals(close=df['close'].values, entries=entries.values,
        sl_stop=sl_stop.values, tp_stop=tp_stop.values, init_cash=INITIAL_CAPITAL,
        fees=FEES, freq='1D', **kwargs)

    # Bereken alle metrics
    # Bereken alle metrics
    metrics = {'total_return': float(pf.total_return()),
        'sharpe_ratio': float(pf.sharpe_ratio()),
        'sortino_ratio': float(pf.sortino_ratio()),
        'calmar_ratio': float(pf.calmar_ratio()),
        'max_drawdown': float(pf.max_drawdown()),
        'win_rate': float(pf.trades.win_rate() if len(pf.trades) > 0 else 0),
        'trades_count': len(pf.trades), 'profit_factor': float(
            pf.trades['pnl'].sum() / abs(
                pf.trades['pnl'][pf.trades['pnl'] < 0].sum()) if len(
                pf.trades[pf.trades['pnl'] < 0]) > 0 else float('inf')),
        'avg_win': float(pf.trades.loc[pf.trades['pnl'] > 0, 'pnl'].mean() if len(
            pf.trades[pf.trades['pnl'] > 0]) > 0 else 0), 'avg_loss': float(
            pf.trades.loc[pf.trades['pnl'] < 0, 'pnl'].mean() if len(
                pf.trades[pf.trades['pnl'] < 0]) > 0 else 0),
        'max_win': float(pf.trades['pnl'].max() if len(pf.trades) > 0 else 0),
        'max_loss': float(pf.trades['pnl'].min() if len(pf.trades) > 0 else 0),
        'avg_duration': float(
            pf.trades['duration'].mean() if len(pf.trades) > 0 else 0), }

    # FTMO check
    compliant, profit_target = check_ftmo_compliance(pf, metrics)
    metrics['ftmo_compliant'] = compliant
    metrics['profit_target_reached'] = profit_target

    # Log belangrijke metrics
    logger.info(f"\n===== BACKTEST RESULTATEN VOOR {strategy_name} OP {symbol} =====")
    logger.info(f"Totaal rendement: {metrics['total_return']:.2%}")
    logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Win rate: {metrics['win_rate']:.2%}")
    logger.info(f"Trades: {metrics['trades_count']}")
    logger.info(f"FTMO compliant: {'JA' if compliant else 'NEE'}")

    # Maak gedetailleerde visualisaties
    timeframe_str = f"_{timeframe}" if timeframe else ""

    # 1. Equity curve
    plt.figure(figsize=(12, 6))
    pf.plot()
    plt.title(f"{strategy_name} Equity Curve - {symbol}")
    equity_file = output_path / f"{strategy_name}_{symbol}{timeframe_str}_equity.png"
    plt.savefig(equity_file)
    plt.close()

    # 2. Drawdown curve
    plt.figure(figsize=(12, 6))
    pf.drawdown().plot()
    plt.title(f"{strategy_name} Drawdowns - {symbol}")
    drawdown_file = output_path / f"{strategy_name}_{symbol}{timeframe_str}_drawdowns.png"
    plt.savefig(drawdown_file)
    plt.close()

    # 3. Maandelijkse returns heatmap
    try:
        plt.figure(figsize=(12, 8))
        returns_monthly = pf.returns().resample('M').sum()
        months = returns_monthly.index.strftime('%Y-%m')

        plt.imshow([returns_monthly.values.flatten()], cmap='RdYlGn', aspect='auto')
        plt.colorbar(label='Returns')
        plt.xticks(range(len(months)), months, rotation=90)
        plt.yticks([])
        plt.title(f"{strategy_name} Monthly Returns - {symbol}")
        monthly_file = output_path / f"{strategy_name}_{symbol}{timeframe_str}_monthly.png"
        plt.savefig(monthly_file)
        plt.close()
    except Exception as e:
        logger.warning(f"Kon geen maandelijkse heatmap maken: {str(e)}")

    # 4. Trade distribution
    if len(pf.trades) > 0:
        plt.figure(figsize=(12, 6))
        pf.trades['pnl'].plot.hist(bins=20, alpha=0.7)
        plt.axvline(0, color='r', linestyle='--')
        plt.title(f"{strategy_name} Trade Distribution - {symbol}")
        trades_file = output_path / f"{strategy_name}_{symbol}{timeframe_str}_trades_dist.png"
        plt.savefig(trades_file)
        plt.close()

    # Sla gedetailleerde trade lijst op
    trades_csv = output_path / f"{strategy_name}_{symbol}{timeframe_str}_trades.csv"
    if len(pf.trades) > 0:
        pf.trades.records_readable.to_csv(trades_csv)

    # Sla alle resultaten op in JSON
    results_file = output_path / f"{strategy_name}_{symbol}{timeframe_str}_results.json"
    with open(results_file, 'w') as f:
        json.dump(
            {'strategy': strategy_name, 'symbol': symbol, 'timeframe': str(timeframe),
                'parameters': parameters, 'metrics': metrics}, f, indent=2)

    logger.info(f"Resultaten en grafieken opgeslagen in {output_path}")

    return pf, metrics


def monte_carlo_analysis(pf, n_simulations=1000):
    """
    Voer een Monte Carlo analyse uit om de robuustheid van de strategie te testen.
    """
    if len(pf.trades) < 10:
        logger.warning("Te weinig trades voor Monte Carlo analyse")
        return None

    # Verzamel trades
    trades = pf.trades.records

    # Maak arrays voor resultaten
    returns = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    sharpe_ratios = np.zeros(n_simulations)

    # Simuleer verschillende trade volgordes
    for i in range(n_simulations):
        # Shuffle de trades (willekeurige volgorde)
        shuffled_indices = np.random.permutation(len(trades))
        shuffled_trades = trades.iloc[shuffled_indices].reset_index(drop=True)

        # Bereken equity curve
        equity = INITIAL_CAPITAL
        equity_curve = [INITIAL_CAPITAL]
        peak = INITIAL_CAPITAL
        drawdowns = []

        for _, trade in shuffled_trades.iterrows():
            equity += trade['pnl']
            equity_curve.append(equity)
            peak = max(peak, equity)
            drawdown = (equity - peak) / peak
            drawdowns.append(drawdown)

        # Bereken metrics
        final_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
        max_drawdown = min(drawdowns) if drawdowns else 0

        # Sharpe ratio (simpele versie)
        equity_array = np.array(equity_curve)
        daily_returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(
            252) if np.std(daily_returns) > 0 else 0

        # Opslaan
        returns[i] = final_return
        max_drawdowns[i] = max_drawdown
        sharpe_ratios[i] = sharpe

    # Bereken statistieken
    results = {'return_mean': float(np.mean(returns)),
        'return_median': float(np.median(returns)),
        'return_std': float(np.std(returns)), 'return_min': float(np.min(returns)),
        'return_max': float(np.max(returns)),
        'return_5pct': float(np.percentile(returns, 5)),
        'return_95pct': float(np.percentile(returns, 95)),

        'max_drawdown_mean': float(np.mean(max_drawdowns)),
        'max_drawdown_median': float(np.median(max_drawdowns)),
        'max_drawdown_std': float(np.std(max_drawdowns)),
        'max_drawdown_worst': float(np.min(max_drawdowns)),
        'max_drawdown_5pct': float(np.percentile(max_drawdowns, 5)),
        'max_drawdown_95pct': float(np.percentile(max_drawdowns, 95)),

        'sharpe_mean': float(np.mean(sharpe_ratios)),
        'sharpe_median': float(np.median(sharpe_ratios)),
        'sharpe_std': float(np.std(sharpe_ratios)), }

    # Plot histogrammen
    output_path = Path(OUTPUT_DIR)

    # Return distribution
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.7)
    plt.axvline(results['return_5pct'], color='r', linestyle='--',
                label='5% Percentile')
    plt.axvline(results['return_mean'], color='g', linestyle='-', label='Mean')
    plt.axvline(results['return_95pct'], color='r', linestyle='--',
                label='95% Percentile')
    plt.title('Monte Carlo: Return Distribution')
    plt.legend()
    plt.savefig(output_path / 'monte_carlo_returns.png')
    plt.close()

    # Drawdown distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_drawdowns, bins=50, alpha=0.7)
    plt.axvline(results['max_drawdown_5pct'], color='r', linestyle='--',
                label='5% Percentile')
    plt.axvline(results['max_drawdown_mean'], color='g', linestyle='-', label='Mean')
    plt.axvline(results['max_drawdown_95pct'], color='r', linestyle='--',
                label='95% Percentile')
    plt.title('Monte Carlo: Max Drawdown Distribution')
    plt.legend()
    plt.savefig(output_path / 'monte_carlo_drawdowns.png')
    plt.close()

    logger.info(f"\n===== MONTE CARLO RESULTATEN ({n_simulations} SIMULATIES) =====")
    logger.info(f"Gemiddeld rendement: {results['return_mean']:.2%}")
    logger.info(
        f"5% - 95% rendement: {results['return_5pct']:.2%} - {results['return_95pct']:.2%}")
    logger.info(f"Gemiddelde max drawdown: {results['max_drawdown_mean']:.2%}")
    logger.info(f"Gemiddelde Sharpe: {results['sharpe_mean']:.2f}")

    return results


def main():
    import argparse
    import MetaTrader5 as mt5

    parser = argparse.ArgumentParser(description="Sophy4 Extended Backtest")
    parser.add_argument("--strategy", type=str, required=True,
                        help="Naam van de strategie")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbool")
    parser.add_argument("--timeframe", type=str,
                        choices=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1'],
                        default='D1', help="Timeframe (default: D1)")
    parser.add_argument("--params_file", type=str,
                        help="JSON file met parameters (optioneel)")
    parser.add_argument("--window", type=int,
                        help="Window parameter (indien geen params_file)")
    parser.add_argument("--std_dev", type=float,
                        help="Std Dev parameter (indien geen params_file)")
    parser.add_argument("--sl_fixed_percent", type=float,
                        help="Stop-loss percentage (indien geen params_file)")
    parser.add_argument("--tp_fixed_percent", type=float,
                        help="Take-profit percentage (indien geen params_file)")
    parser.add_argument("--monte_carlo", action="store_true",
                        help="Voer Monte Carlo analyse uit")

    args = parser.parse_args()

    # Timeframe vertalen naar MT5 constante
    tf_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1}
    timeframe = tf_map.get(args.timeframe, mt5.TIMEFRAME_D1)

    # Bepaal parameters
    if args.params_file:
        with open(args.params_file, 'r') as f:
            data = json.load(f)
            if 'top_results' in data and len(data['top_results']) > 0:
                parameters = data['top_results'][0]['params']
            else:
                parameters = data['parameters']
    else:
        parameters = {}
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

    # Voer backtest uit
    pf, metrics = run_extended_backtest(args.strategy, parameters, args.symbol,
                                        timeframe)

    # Monte Carlo analyse
    if args.monte_carlo and pf is not None:
        monte_carlo_analysis(pf)


if __name__ == "__main__":
    main()