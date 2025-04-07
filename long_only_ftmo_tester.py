"""
Sophy3 - Long-Only Bollinger Breakout Tester
Functie: Bollinger Band breakout strategie tester (alleen long signalen)
Auteur: AI Assistant
Laatste update: 2025-04-07

Gebruik:
  python long_only_breakout.py --symbol GER40.cash --exact_dates
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt

# Stel de frequentie globaal in voor VectorBT
vbt.settings.array_wrapper['freq'] = '5m'  # Dagelijkse data frequentie

# Configuratie voor logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()])
logger = logging.getLogger()

# FTMO limieten
MAX_DAILY_LOSS = 0.05  # 5% maximaal dagelijks verlies
MAX_TOTAL_LOSS = 0.10  # 10% maximaal totale drawdown
PROFIT_TARGET = 0.10  # 10% winstdoelstelling


def calculate_atr(data, window=14):
    """Calculate Average True Range"""
    high = data['high']
    low = data['low']
    close = data['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=window).mean()

    return atr


def test_long_only_strategy(symbol='GER40.cash', window=50, std_dev=2.0,
        sl_method='fixed_percent', tp_method='atr_based', sl_fixed=0.02, tp_fixed=0.03,
        sl_atr_mult=2.0, tp_atr_mult=3.0, period_days=1095, initial_capital=10000.0,
        output_dir='results', end_date=None, show_trades=True, exact_dates=False,
        fees=0.0002  # Pas dit aan om commissie en slippage te wijzigen
):
    """
    Test een Bollinger Band breakout strategie (alleen LONG signalen) voor FTMO compliance.

    Args:
        symbol: Trading symbool (bv. 'GER40.cash')
        window: Bollinger Band window periode
        std_dev: Aantal standaarddeviaties voor BB
        sl_method: 'fixed_percent' of 'atr_based'
        tp_method: 'fixed_percent' of 'atr_based'
        sl_fixed: Vaste stop loss percentage (als sl_method='fixed_percent')
        tp_fixed: Vaste take profit percentage (als tp_method='fixed_percent')
        sl_atr_mult: ATR vermenigvuldiger voor stop loss (als sl_method='atr_based')
        tp_atr_mult: ATR vermenigvuldiger voor take profit (als tp_method='atr_based')
        period_days: Aantal dagen voor backtest
        initial_capital: Startkapitaal
        output_dir: Map voor resultaten
        end_date: Einddatum voor de backtest (standaard: nu)
        show_trades: Toon alle trades in rapport
        exact_dates: Gebruik specifieke datumrange voor vergelijking met originele test
        fees: Commissie + slippage (als percentage, 0.0002 = 0.02%)

    Returns:
        dict: Testresultaten en FTMO aanbeveling
    """
    # Maak output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 1. Initialiseer MT5
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt: {mt5.last_error()}")
        return {'success': False, 'error': 'MT5 initialisatie mislukt'}

    logger.info(
        f"Start LONG-ONLY test voor {symbol} met window={window}, std_dev={std_dev}")

    # 2. Stel datumrange in
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Ongeldige datumformat voor end_date. Gebruik YYYY-MM-DD")
            return {'success': False, 'error': 'Ongeldige datumformat'}

    if exact_dates:
        # Gebruik exact dezelfde periode als in de originele test
        start_date = datetime(2022, 4, 11)
        end_date = datetime(2025, 4, 4)
    else:
        start_date = end_date - timedelta(days=period_days)

    # 3. Haal historische data op
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start_date, end_date)
    if rates is None or len(rates) == 0:
        logger.error(f"Geen data ontvangen voor {symbol}")
        return {'success': False, 'error': 'Geen data ontvangen'}

    # Converteer naar DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    logger.info(f"Data geladen: {len(df)} rijen van {df.index[0]} tot {df.index[-1]}")

    # 4. Bereken Bollinger Bands
    df['sma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['sma'] + (std_dev * df['std'])
    df['lower_band'] = df['sma'] - (std_dev * df['std'])

    # 5. Bereken ATR voor dynamische stops
    df['atr'] = calculate_atr(df, window=14)

    # 6. Genereer ALLEEN LONG trading signalen
    df['entries'] = df['close'] > df[
        'upper_band']  # ALLEEN triggers boven de upper band

    logger.info(f"Aantal LONG entry signalen gegenereerd: {df['entries'].sum()}")

    # 7. Stop loss en take profit
    if sl_method == 'atr_based':
        df['sl_stop'] = sl_atr_mult * df['atr'] / df['close']
    else:  # fixed_percent
        df['sl_stop'] = sl_fixed

    if tp_method == 'atr_based':
        df['tp_stop'] = tp_atr_mult * df['atr'] / df['close']
    else:  # fixed_percent
        df['tp_stop'] = tp_fixed

    # 8. Voer backtest uit - ALLEEN LONG posities
    pf = vbt.Portfolio.from_signals(close=df['close'], entries=df['entries'],
        exits=None,  # We gebruiken SL/TP voor exits
        # direction niet nodig voor long-only (standaard is long)
        sl_stop=df['sl_stop'], tp_stop=df['tp_stop'], size=None,  # Auto-size
        size_type='value', init_cash=initial_capital, freq='1D',
        # Expliciet specificeren
        fees=fees  # Commissie + slippage
    )

    # 9. Bereken performance metrics
    metrics = {'total_return': pf.total_return(), 'sharpe_ratio': pf.sharpe_ratio(),
        'max_drawdown': pf.max_drawdown(),
        'win_rate': pf.trades.win_rate() if len(pf.trades) > 0 else 0,
        'trades_count': len(pf.trades)}

    logger.info(f"Backtest resultaten voor {symbol} (LONG-ONLY):")
    logger.info(
        f"  Total Return: {metrics['total_return']:.4f} ({metrics['total_return'] * 100:.2f}%)")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(
        f"  Max Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown'] * 100:.2f}%)")
    logger.info(
        f"  Win Rate: {metrics['win_rate']:.4f} ({metrics['win_rate'] * 100:.2f}%)")
    logger.info(f"  Trades: {metrics['trades_count']}")

    # 10. FTMO compliance check
    daily_returns = pf.returns()
    max_daily_loss = daily_returns.min()
    daily_loss_violated = abs(max_daily_loss) > MAX_DAILY_LOSS
    drawdown_violated = metrics['max_drawdown'] > MAX_TOTAL_LOSS
    profit_target_reached = metrics['total_return'] >= PROFIT_TARGET
    overall_compliant = not (daily_loss_violated or drawdown_violated)

    logger.info(f"FTMO compliance check:")
    logger.info(
        f"  Dagelijkse verliesregel overtreden: {'JA' if daily_loss_violated else 'NEE'}")
    logger.info(
        f"  Maximale drawdown regel overtreden: {'JA' if drawdown_violated else 'NEE'}")
    logger.info(f"  Algemene compliance: {'GOED' if overall_compliant else 'SLECHT'}")

    # 11. Visualisatie van resultaten
    try:
        fig, axs = plt.subplots(2, 1, figsize=(14, 10),
                                gridspec_kw={'height_ratios': [2, 1]})

        # Plot price chart
        axs[0].plot(df.index, df['close'], 'b-', linewidth=1, label='Close')
        axs[0].plot(df.index, df['upper_band'], 'r--', linewidth=1, label='Upper Band')
        axs[0].plot(df.index, df['lower_band'], 'g--', linewidth=1,
                    label='Lower Band (niet gebruikt)')

        # Mark entry signals
        entry_points = df[df['entries']].index
        if len(entry_points) > 0:
            axs[0].scatter(entry_points, df.loc[entry_points, 'close'], marker='^',
                           color='green', s=100, label='Long Entry')

        axs[0].set_title('Price Chart with Bollinger Bands (LONG-ONLY)')
        axs[0].set_ylabel('Price')
        axs[0].legend()
        axs[0].grid(True)

        # Plot equity curve
        equity = pf.equity_curve
        axs[1].plot(equity.index, equity.iloc[:, 0], 'b-', linewidth=1.5)
        axs[1].set_title('Portfolio Equity Curve (LONG-ONLY)')
        axs[1].set_ylabel('Portfolio Value')
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(output_path / f"{symbol}_long_only_performance.png", dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
        logger.info(
            f"Performance grafiek opgeslagen in {output_path / f'{symbol}_long_only_performance.png'}")
    except Exception as e:
        logger.error(f"Fout bij visualisatie: {e}")
        logger.info("Visualisatie overgeslagen, maar analyse is voltooid")

    # 12. Aanbeveling voor FTMO
    safe_for_ftmo = (overall_compliant and metrics['sharpe_ratio'] > 1.0 and metrics[
        'total_return'] > 0 and metrics['win_rate'] > 0.5)

    recommendation = (
        "Deze LONG-ONLY strategie lijkt veilig te gebruiken op een FTMO account." if safe_for_ftmo else "Deze LONG-ONLY strategie heeft aanpassingen nodig voordat het veilig is voor FTMO gebruik.")

    logger.info(f"AANBEVELING: {recommendation}")

    # 13. Genereer samenvatting rapport
    try:
        summary_path = output_path / f"{symbol}_long_only_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"====== SOPHY3 LONG-ONLY STRATEGIE TEST RAPPORT ======\n")
            f.write(f"Symbool: {symbol}\n")
            f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Test periode: {df.index[0].strftime('%Y-%m-%d')} tot {df.index[-1].strftime('%Y-%m-%d')}\n\n")

            f.write(f"--- STRATEGIE PARAMETERS ---\n")
            f.write(f"Trading Richting: ALLEEN LONG\n")
            f.write(f"Window: {window}\n")
            f.write(f"Std Dev: {std_dev}\n")
            f.write(f"Stop Loss Methode: {sl_method}\n")
            f.write(f"Take Profit Methode: {tp_method}\n")
            if sl_method == 'fixed_percent':
                f.write(f"Stop Loss Percentage: {sl_fixed * 100:.2f}%\n")
            else:
                f.write(f"Stop Loss ATR Multiplier: {sl_atr_mult}\n")
            if tp_method == 'fixed_percent':
                f.write(f"Take Profit Percentage: {tp_fixed * 100:.2f}%\n")
            else:
                f.write(f"Take Profit ATR Multiplier: {tp_atr_mult}\n")
            f.write(f"Commissie + Slippage: {fees * 100:.4f}%\n\n")

            f.write(f"--- BACKTEST RESULTATEN ---\n")
            f.write(f"Totaal Rendement: {metrics['total_return'] * 100:.2f}%\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%\n")
            f.write(f"Win Rate: {metrics['win_rate'] * 100:.2f}%\n")
            f.write(f"Totaal Trades: {metrics['trades_count']}\n\n")

            f.write(f"--- FTMO COMPLIANCE CHECK ---\n")
            f.write(
                f"Dagelijkse verliesregel overtreden: {'JA' if daily_loss_violated else 'NEE'}\n")
            f.write(
                f"Maximale drawdown regel overtreden: {'JA' if drawdown_violated else 'NEE'}\n")
            f.write(
                f"Winstdoelstelling bereikt: {'JA' if profit_target_reached else 'NEE'}\n")
            f.write(
                f"Algemene compliance: {'GOED' if overall_compliant else 'SLECHT'}\n\n")

            if show_trades and len(pf.trades) > 0:
                f.write(f"--- TRADES OVERZICHT ---\n")
                trades_df = pf.trades.records
                for i, trade in enumerate(trades_df):
                    entry_date = pd.Timestamp(trade['entry_time'])
                    exit_date = pd.Timestamp(trade['exit_time'])
                    entry_price = trade['entry_price']
                    exit_price = trade['exit_price']
                    pnl = trade['pnl']
                    pnl_pct = trade['return']
                    status = "WIN" if pnl > 0 else "LOSS"
                    f.write(
                        f"Trade {i + 1}: LONG {entry_date.strftime('%Y-%m-%d')} ({entry_price:.2f}) -> {exit_date.strftime('%Y-%m-%d')} ({exit_price:.2f}) | {pnl:.2f} ({pnl_pct * 100:.2f}%) | {status}\n")
                f.write("\n")

            f.write(f"--- AANBEVELING ---\n")
            f.write(f"{recommendation}\n\n")

            f.write(f"Gegenereerd door Sophy3 Test Framework (LONG-ONLY versie)\n")

        logger.info(f"Samenvattingsrapport opgeslagen in {summary_path}")
    except Exception as e:
        logger.error(f"Fout bij genereren van samenvattingsrapport: {e}")

    # 14. Eindresultaat
    return {'success': True, 'metrics': metrics, 'ftmo_compliant': overall_compliant,
        'recommendation': recommendation, 'safe_for_ftmo': safe_for_ftmo}


def main():
    """Main functie om het script te draaien."""
    parser = argparse.ArgumentParser(
        description='Sophy3 LONG-ONLY Bollinger Breakout Strategie Tester')

    parser.add_argument('--symbol', type=str, default='GER40.cash',
                        help='Symbol to test')
    parser.add_argument('--window', type=int, default=50,
                        help='Bollinger Band window (default: 50)')
    parser.add_argument('--std_dev', type=float, default=2.0,
                        help='Bollinger Band std dev (default: 2.0)')
    parser.add_argument('--sl_method', type=str, choices=['atr_based', 'fixed_percent'],
                        default='fixed_percent',
                        help='Stop loss method (default: fixed_percent)')
    parser.add_argument('--tp_method', type=str, choices=['atr_based', 'fixed_percent'],
                        default='atr_based',
                        help='Take profit method (default: atr_based)')
    parser.add_argument('--sl_fixed', type=float, default=0.02,
                        help='Fixed stop loss percentage (default: 0.02)')
    parser.add_argument('--tp_fixed', type=float, default=0.03,
                        help='Fixed take profit percentage (default: 0.03)')
    parser.add_argument('--sl_atr_mult', type=float, default=2.0,
                        help='Stop loss ATR multiplier (default: 2.0)')
    parser.add_argument('--tp_atr_mult', type=float, default=3.0,
                        help='Take profit ATR multiplier (default: 3.0)')
    parser.add_argument('--period', type=int, default=1095,
                        help='Test period in days (default: 1095)')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--end_date', type=str,
                        help='End date for backtest (format: YYYY-MM-DD)')
    parser.add_argument('--exact_dates', action='store_true',
                        help='Use exact date range from original test (Apr 11, 2022 to Apr 4, 2025)')
    parser.add_argument('--show_trades', action='store_true',
                        help='Show all trades in report')
    parser.add_argument('--fees', type=float, default=0.0002,
                        help='Fees (commission + slippage) as percentage (default: 0.0002 = 0.02%)')

    args = parser.parse_args()

    # Run test
    results = test_long_only_strategy(symbol=args.symbol, window=args.window,
        std_dev=args.std_dev, sl_method=args.sl_method, tp_method=args.tp_method,
        sl_fixed=args.sl_fixed, tp_fixed=args.tp_fixed, sl_atr_mult=args.sl_atr_mult,
        tp_atr_mult=args.tp_atr_mult, period_days=args.period,
        initial_capital=args.capital, output_dir=args.output_dir,
        end_date=args.end_date, show_trades=args.show_trades,
        exact_dates=args.exact_dates, fees=args.fees)

    if results['success']:
        print(f"\n====== SAMENVATTING VOOR {args.symbol} (LONG-ONLY) ======")
        print(f"Strategie Parameters: Window={args.window}, StdDev={args.std_dev}")
        print(f"Backtest Rendement: {results['metrics']['total_return'] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.4f}")
        print(f"FTMO Compliance: {'GOED' if results['ftmo_compliant'] else 'SLECHT'}")
        print(f"AANBEVELING: {results['recommendation']}")
        print(
            f"\nGedetailleerde resultaten en grafieken zijn opgeslagen in: {args.output_dir}/")
    else:
        print(f"Test mislukt: {results.get('error', 'Onbekende fout')}")


if __name__ == "__main__":
    main()