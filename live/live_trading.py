# Sophy4/live/live_trading.py
import time
from typing import Dict, Union, Optional, List

import MetaTrader5 as mt5
import pandas as pd

from config import logger


def run_live_trading(df: pd.DataFrame, symbol: str, use_trailing_stop: bool = False,
                     risk_per_trade: float = 0.01, max_positions: int = 3) -> Dict[str, Union[bool, float, str]]:
    """Voer live trading uit op basis van signalen."""
    result = {"success": False, "trade_executed": False, "symbol": symbol,
        "timestamp": pd.Timestamp.now(), "message": "", "ticket": None}

    try:
        if df.empty:
            result["message"] = "Geen data beschikbaar"
            return result

        if not mt5.initialize():
            result["message"] = "MT5 initialisatie mislukt"
            return result

        # Haal symbol info op aan het begin
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            result["message"] = f"Kan symbool info niet ophalen voor {symbol}"
            return result

        # Controleer huidige posities
        positions = check_positions([symbol])
        if len(positions) >= max_positions:
            result["message"] = f"Max posities ({max_positions}) bereikt voor {symbol}"
            logger.warning(result["message"])
            return result

        latest_data = df.iloc[-1]
        if not latest_data.get('entries', False):
            result["message"] = "Geen entry signaal"
            return result

        # Gebruik realtime ask-prijs
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            result["message"] = f"Kan realtime prijs niet ophalen voor {symbol}"
            return result
        current_price = tick.ask

        # Bereken SL en TP met juiste precisie
        sl_percent = latest_data['sl_stop']
        tp_percent = latest_data['tp_stop']
        sl_price = round(current_price * (1 - sl_percent), symbol_info.digits)
        tp_price = round(current_price * (1 + tp_percent), symbol_info.digits)

        # Positiegrootte
        account_info = mt5.account_info()
        if account_info is None:
            result["message"] = "Kan account info niet ophalen"
            return result

        balance = account_info.balance
        risk_amount = balance * risk_per_trade
        price_risk = current_price - sl_price

        contract_size = symbol_info.trade_contract_size
        lot_step = symbol_info.volume_step
        calculated_size = risk_amount / (price_risk * contract_size) if price_risk > 0 else symbol_info.volume_min
        size = max(round(calculated_size / lot_step) * lot_step, symbol_info.volume_min)

        # Voer trade uit
        portfolio_kwargs = {
            'close': df['close'],
            'entries': df['entries'],
            'sl_stop': df['sl_stop'],
            'tp_stop': df['tp_stop'],
            'init_cash': account_info.balance,
            'fees': 0.0002,  # Standaard commissie
            'freq': '1D'
        }

        trade_result = execute_trade(symbol, current_price, sl_price, tp_price, size)
        if trade_result:
            result["success"] = True
            result["trade_executed"] = True
            result["message"] = f"Trade uitgevoerd: {symbol} @ {current_price}, SL={sl_price}, TP={tp_price}, Size={size}"
            result.update({"entry_price": current_price, "sl_price": sl_price,
                "tp_price": tp_price, "position_size": size, "risk_amount": risk_amount,
                "ticket": mt5.positions_get(symbol=symbol)[0].ticket if mt5.positions_get(symbol=symbol) else None})
            logger.info(f"LIVE TRADE: {result['message']}")

            # Trailing stop logica (handmatig)
            if use_trailing_stop:
                result["trailing_active"] = False
                logger.info("Trailing stop geactiveerd (handmatige controle vereist)")

        else:
            result["message"] = "Trade uitvoering mislukt"

        return result

    except Exception as e:
        result["message"] = f"Fout tijdens live trading: {str(e)}"
        logger.exception("Live trading fout")
        return result


def execute_trade(symbol: str, price: float, sl: float, tp: float, size: float = 1.0,
                  retry_attempts: int = 3, retry_delay: float = 2.0) -> bool:
    """Voer een trade uit via MetaTrader 5."""
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt: {mt5.last_error()}")
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbool {symbol} niet gevonden in MT5")
        return False

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)
        logger.info(f"Symbool {symbol} toegevoegd aan MarketWatch")

    request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": size,
        "type": mt5.ORDER_TYPE_BUY, "price": price, "sl": sl, "tp": tp, "magic": 123456,
        "comment": "Sophy4_Bollong", "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC}

    for attempt in range(1, retry_attempts + 1):
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"Trade uitgevoerd: {symbol} @ {price}, SL={sl}, TP={tp}, Volume={size}, Ticket={result.order}")
            return True
        elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
            new_price = mt5.symbol_info_tick(symbol).ask
            logger.warning(f"Requote: nieuwe prijs {new_price} (was {price})")
            request["price"] = new_price
        else:
            logger.error(
                f"Trade mislukt (poging {attempt}): {result.comment}, code: {result.retcode}")
            if attempt < retry_attempts:
                time.sleep(retry_delay)

    return False


def check_positions(symbols: List[str] = None) -> List[Dict]:
    """Controleer open posities in MT5."""
    if not mt5.initialize():
        logger.error("MT5 initialisatie mislukt")
        return []

    positions = mt5.positions_get() if not symbols else sum(
        (mt5.positions_get(symbol=s) or [] for s in symbols), [])
    if not positions:
        return []

    position_info = []
    for pos in positions:
        pnl_percent = ((pos.price_current / pos.price_open) - 1) * 100 if pos.type == 0 else (
                (pos.price_open / pos.price_current) - 1) * 100
        position_info.append({"symbol": pos.symbol, "ticket": pos.ticket,
            "type": "BUY" if pos.type == 0 else "SELL", "volume": pos.volume,
            "open_price": pos.price_open, "current_price": pos.price_current,
            "sl": pos.sl, "tp": pos.tp, "pnl": pos.profit,
            "pnl_percent": round(pnl_percent, 2), "swap": pos.swap,
            "time": pd.Timestamp.fromtimestamp(pos.time), "comment": pos.comment,
            "magic": pos.magic})
    return position_info


def modify_position(ticket: int, sl: Optional[float] = None,
                    tp: Optional[float] = None) -> bool:
    """Wijzig een bestaande positie."""
    if not mt5.initialize():
        logger.error("MT5 initialisatie mislukt")
        return False

    position = mt5.positions_get(ticket=ticket)
    if not position:
        logger.error(f"Positie met ticket {ticket} niet gevonden")
        return False

    position = position[0]
    request = {"action": mt5.TRADE_ACTION_SLTP, "symbol": position.symbol,
        "position": ticket, "magic": position.magic,
        "sl": sl if sl is not None else position.sl,
        "tp": tp if tp is not None else position.tp}

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"Positie wijziging mislukt: {result.comment}, code: {result.retcode}")
        return False

    logger.info(f"Positie {ticket} gewijzigd: SL={sl}, TP={tp}")
    return True


def close_position(ticket: int) -> bool:
    """Sluit een bestaande positie."""
    if not mt5.initialize():
        logger.error("MT5 initialisatie mislukt")
        return False

    position = mt5.positions_get(ticket=ticket)
    if not position:
        logger.error(f"Positie met ticket {ticket} niet gevonden")
        return False

    position = position[0]
    order_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask

    request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": position.symbol,
        "volume": position.volume, "type": order_type, "position": ticket,
        "price": price, "magic": position.magic, "comment": "Sophy4_Close",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"Positie sluiten mislukt: {result.comment}, code: {result.retcode}")
        return False

    logger.info(f"Positie {ticket} gesloten: {position.symbol} @ {price}")
    return True


def manage_trailing_stop(position: Dict, trailing_stop_percent: float,
                         current_price: float) -> bool:
    """Handmatig beheren van een trailing stop."""
    if position['type'] != "BUY":
        logger.warning("Trailing stop alleen ondersteund voor BUY-posities")
        return False

    entry_price = position['open_price']
    profit_percent = (current_price - entry_price) / entry_price
    if profit_percent < 0:
        return False  # Geen trailing stop als prijs onder entry ligt

    new_sl = current_price * (1 - trailing_stop_percent)
    if new_sl > position['sl'] and new_sl > entry_price:
        return modify_position(position['ticket'], sl=new_sl)
    return False