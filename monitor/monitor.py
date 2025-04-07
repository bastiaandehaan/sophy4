# Sophy4/monitor/monitor.py
from config import logger, OUTPUT_DIR

def monitor_performance(pf=None, live=False):
    if live:
        logger.info("Live monitoring: Nog niet volledig geïmplementeerd")
        # Hier kun je MT5 posities checken in de toekomst
    elif pf:
        equity = pf.value()
        logger.info(f"Huidige Equity: {equity.iloc[-1]:.2f}")
        with open(OUTPUT_DIR / "monitor_log.txt", 'a') as f:
            f.write(f"{equity.index[-1]}: Equity={equity.iloc[-1]:.2f}\n")