# Sophy4/backtest/backtest.py
import vectorbt as vbt
import matplotlib.pyplot as plt
from config import INITIAL_CAPITAL, FEES, logger, OUTPUT_DIR


def run_backtest(df, symbol):
    pf = vbt.Portfolio.from_signals(close=df['close'], entries=df['entries'],
        sl_stop=df['sl_stop'], tp_stop=df['tp_stop'], init_cash=INITIAL_CAPITAL,
        fees=FEES, freq='1D')
    metrics = {'total_return': pf.total_return(), 'sharpe_ratio': pf.sharpe_ratio(),
        'max_drawdown': pf.max_drawdown(),
        'win_rate': pf.trades.win_rate() if len(pf.trades) > 0 else 0,
        'trades_count': len(pf.trades)}
    logger.info(
        f"Backtest Resultaten: Return={metrics['total_return']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}, "
        f"Max Drawdown={metrics['max_drawdown']:.2%}, Win Rate={metrics['win_rate']:.2%}")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    pf.value().plot()
    plt.title(f"{symbol} Equity Curve")
    plt.savefig(OUTPUT_DIR / f"{symbol}_equity.png")
    plt.close()

    return pf, metrics