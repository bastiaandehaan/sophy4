# Sophy4/risk/risk_management.py
from config import INITIAL_CAPITAL, logger

def calculate_position_size(capital, price, sl_percent, max_risk=0.01):
    risk_per_trade = capital * max_risk
    position_size = risk_per_trade / (price * sl_percent)
    logger.info(f"Position Size: {position_size:.2f} units (Risk={max_risk*100}%)")
    return position_size