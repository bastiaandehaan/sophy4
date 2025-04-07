# Sophy4/main.py
from config import SYMBOL, logger
from utils.data_utils import fetch_historical_data, fetch_live_data
from strategies.bollong import bollong_signals
from backtest.backtest import run_backtest
from live.live_trading import run_live_trading
from risk.risk_management import calculate_position_size
from monitor.monitor import monitor_performance
from ftmo_compliance.ftmo_check import check_ftmo_compliance

def main(mode="backtest"):
    if mode == "backtest":
        df = fetch_historical_data(SYMBOL)
        if df is None:
            return
        entries, sl_stop, tp_stop = bollong_signals(df)
        df['entries'] = entries
        df['sl_stop'] = sl_stop
        df['tp_stop'] = tp_stop
        pf, metrics = run_backtest(df, SYMBOL)
        compliant, profit_reached = check_ftmo_compliance(pf, metrics)
        monitor_performance(pf)
    elif mode == "live":
        df = fetch_live_data(SYMBOL)
        if df is None:
            return
        entries, sl_stop, tp_stop = bollong_signals(df)
        df['entries'] = entries
        df['sl_stop'] = sl_stop
        df['tp_stop'] = tp_stop
        run_live_trading(df, SYMBOL)
        size = calculate_position_size(10000, df['close'].iloc[-1], df['sl_stop'].iloc[-1])
        monitor_performance(live=True)

if __name__ == "__main__":
    main(mode="backtest")  # Verander naar "live" voor live trading