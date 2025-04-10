# config.py
import logging
from pathlib import Path

import MetaTrader5 as mt5  # Al aanwezig voor PIP_VALUE

# Logging configuratie
LOG_DIR: Path = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE: Path = LOG_DIR / "sophy4_backtest.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,
    filemode='a'
)
logger: logging.Logger = logging.getLogger()

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

# FTMO limieten
MAX_DAILY_LOSS: float = 0.05  # 5%
MAX_TOTAL_LOSS: float = 0.10  # 10%
PROFIT_TARGET: float = 0.10  # 10%

# Standaardparameters
SYMBOL: str = "GER40.cash"
FEES: float = 0.00005  # Spread-gebaseerd (0.005%)

# Dynamisch berekenen van INITIAL_CAPITAL
def get_initial_capital() -> float:
    """
    Haal de accountbalans op via MT5 om INITIAL_CAPITAL in te stellen.
    Valt terug op een standaardwaarde als MT5 niet beschikbaar is.

    Returns:
        De accountbalans in de accountvaluta.
    """
    try:
        if not mt5.initialize():
            logger.warning("MT5 initialisatie mislukt, gebruik standaard kapitaal.")
            return 10000.0  # Fallback naar standaardwaarde

        account_info = mt5.account_info()
        if account_info is None:
            logger.warning("Kan accountinfo niet ophalen, gebruik standaard kapitaal.")
            return 10000.0

        balance: float = account_info.balance
        logger.info(f"Accountbalans (INITIAL_CAPITAL): {balance}")
        return balance

    except Exception as e:
        logger.error(f"Fout bij ophalen accountbalans: {str(e)}")
        return 10000.0  # Fallback
    finally:
        mt5.shutdown()

INITIAL_CAPITAL: float = get_initial_capital()  # Dynamisch berekend

# Dynamisch berekenen van PIP_VALUE
def get_pip_value(symbol: str) -> float:
    """
    Haal de pip-waarde op voor een gegeven symbool via MT5.
    Valt terug op een standaardwaarde als MT5 niet beschikbaar is.

    Args:
        symbol: Het symbool (bijv. "GER40.cash").

    Returns:
        De pip-waarde in de accountvaluta.
    """
    try:
        if not mt5.initialize():
            logger.warning("MT5 initialisatie mislukt, gebruik standaard pip-waarde.")
            return 1.0  # Fallback voor GER40.cash

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"Kan symboolinfo voor {symbol} niet ophalen, gebruik standaard pip-waarde.")
            return 1.0

        pip_value = symbol_info.point * symbol_info.trade_contract_size
        logger.info(f"Pip-waarde voor {symbol}: {pip_value}")
        return pip_value

    except Exception as e:
        logger.error(f"Fout bij ophalen pip-waarde: {str(e)}")
        return 1.0  # Fallback
    finally:
        mt5.shutdown()

PIP_VALUE: float = get_pip_value(SYMBOL)  # Dynamisch berekend

OUTPUT_DIR: Path = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)