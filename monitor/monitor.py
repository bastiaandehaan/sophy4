# Sophy4/monitor/monitor.py
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import vectorbt as vbt
import MetaTrader5 as mt5

from config import logger, OUTPUT_DIR
from live.live_trading import check_positions, manage_trailing_stop


def monitor_performance(pf=None, live=False, symbols=None, time_filter=False,
                        trading_hours=(9, 17), trailing_stop_percent=0.015):
    """Monitor prestaties van backtest of live trades."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)
    summary = {"timestamp": pd.Timestamp.now(), "status": "running", "metrics": {}}

    try:
        if live:
            if not mt5.initialize():
                raise Exception("MT5 initialisatie mislukt")

            positions = check_positions(symbols)
            if time_filter:
                start, end = trading_hours
                hour = pd.Timestamp.now().hour
                if not ((start <= end and start <= hour <= end) or (
                        start > end and (hour >= start or hour <= end))):
                    positions = []
                    logger.info(f"Buiten handelsuren ({start}-{end})")

            if not positions:
                summary["metrics"] = {"open_positions": 0}
                logger.info("Geen open posities")
            else:
                for pos in positions:
                    if pos["comment"] == "Sophy4_Bollong":
                        tick = mt5.symbol_info_tick(pos["symbol"])
                        if tick and manage_trailing_stop(pos, trailing_stop_percent,
                                                         tick.ask):
                            logger.info(f"Trailing stop aangepast voor {pos['ticket']}")

                total_pnl = sum(pos["pnl"] for pos in positions)
                equity = mt5.account_info().equity if mt5.account_info() else 0
                summary["metrics"] = {"open_positions": len(positions),
                                      "total_pnl": round(total_pnl, 2),
                                      "equity": round(equity, 2)}
                logger.info(
                    f"Live: {len(positions)} posities, P&L: {total_pnl:.2f}, Equity: {equity:.2f}")

                with open(output_path / "monitor_log.txt", 'a') as f:
                    f.write(
                        f"{summary['timestamp']}: P&L={total_pnl:.2f}, Equity={equity:.2f}\n")

                plt.figure(figsize=(10, 6))
                plt.bar(range(len(positions)), [pos["pnl"] for pos in positions],
                        label="P&L")
                plt.axhline(0, color='r', linestyle='--')
                plt.title(f"Live P&L ({summary['timestamp']})")
                plt.legend()
                plt.savefig(output_path / "live_pnl_plot.png")
                plt.close()

        elif pf:
            equity = pf.value()
            metrics = {"equity": round(equity.iloc[-1], 2),
                "total_return": round(pf.total_return() * 100, 2),
                "sharpe_ratio": round(pf.sharpe_ratio(), 2),
                "max_drawdown": round(pf.max_drawdown() * 100, 2),
                "win_rate": round(pf.trades.win_rate() * 100, 2) if len(
                    pf.trades) > 0 else 0}
            summary["metrics"] = metrics
            logger.info(
                f"Backtest: Equity={metrics['equity']:.2f}, Return={metrics['total_return']}%")

            with open(output_path / "monitor_log.txt", 'a') as f:
                f.write(
                    f"{equity.index[-1]}: Equity={metrics['equity']:.2f}, Return={metrics['total_return']}%\n")

            plt.figure(figsize=(10, 6))
            equity.plot(label="Equity")
            plt.title("Backtest Equity")
            plt.legend()
            plt.savefig(output_path / "backtest_equity_plot.png")
            plt.close()

        else:
            raise Exception("Geen input opgegeven")

    except Exception as e:
        summary["status"] = "error"
        summary["message"] = str(e)
        logger.error(f"Monitoring mislukt: {e}")

    return summary


def run_monitor_loop(pf=None, live=False, symbols=None, interval=60, duration=None,
                     time_filter=False, trading_hours=(9, 17),
                     trailing_stop_percent=0.015):
    """Voer een continue monitoring loop uit."""
    start = time.time()
    while duration is None or (time.time() - start) < duration:
        t = time.time()
        summary = monitor_performance(pf, live, symbols, time_filter, trading_hours,
                                      trailing_stop_percent)
        if summary["status"] == "error" and "MT5" in summary.get("message", ""):
            time.sleep(300)  # Wacht 5 min bij MT5-fout
            continue
        time.sleep(max(0, interval - (time.time() - t)))
    logger.info("Monitoring gestopt")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        run_monitor_loop(live=True, symbols=["EURUSD"], interval=60, duration=3600,
                         time_filter=True)
    else:
        logger.info("Backtest monitoring vereist een Portfolio-object")