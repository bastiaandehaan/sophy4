# Sophy4/live/live_trading.py
import MetaTrader5 as mt5

from config import logger


def execute_trade(symbol, price, sl, tp, size=1.0):
    if not mt5.initialize():
        logger.error("MT5 initialisatie mislukt")
        return False
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 123456,
        "comment": "Bollong Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Trade mislukt: {result.comment}")
        return False
    logger.info(f"Trade uitgevoerd: {symbol} @ {price}, SL={sl}, TP={tp}")
    return True

def run_live_trading(df, symbol):
    latest_data = df.iloc[-1]
    if latest_data['entries']:
        sl = latest_data['close'] * (1 - latest_data['sl_stop'])
        tp = latest_data['close'] * (1 + latest_data['tp_stop'])
        execute_trade(symbol, latest_data['close'], sl, tp)