# Sophy4/ftmo_compliance/ftmo_check.py
from config import MAX_DAILY_LOSS, MAX_TOTAL_LOSS, PROFIT_TARGET, logger

def check_ftmo_compliance(pf, metrics):
    daily_returns = pf.returns()
    daily_loss_violated = abs(daily_returns.min()) > MAX_DAILY_LOSS
    drawdown_violated = metrics['max_drawdown'] > MAX_TOTAL_LOSS
    profit_target_reached = metrics['total_return'] >= PROFIT_TARGET
    compliant = not (daily_loss_violated or drawdown_violated)
    logger.info(f"FTMO Compliance: {'GOED' if compliant else 'SLECHT'}")
    logger.info(f"Profit Target bereikt: {'JA' if profit_target_reached else 'NEE'}")
    return compliant, profit_target_reached