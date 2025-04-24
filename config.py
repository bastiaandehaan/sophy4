import logging
from pathlib import Path
import MetaTrader5 as mt5

# Logging configuration
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

# FTMO limits
MAX_DAILY_LOSS: float = 0.05  # 5%
MAX_TOTAL_LOSS: float = 0.10  # 10%
PROFIT_TARGET: float = 0.10  # 10%

# Trading symbols
SYMBOLS: list = ["EURUSD", "GER40.cash", "US30.cash", "XAUUSD"]

# Correlated symbols for risk management
CORRELATED_SYMBOLS: dict = {
    'EURUSD': ['GBPUSD', 'USDCHF', 'EURGBP'],
    'XAUUSD': ['XAGUSD', 'USDCAD'],
    'GER40.cash': ['FRA40.cash', 'UK100.cash'],
    'US30.cash': ['US500.cash', 'NAS100.cash']
}

# Default parameters
FEES: float = 0.00005  # Spread-based (0.005%)

# Dynamically calculate INITIAL_CAPITAL
def get_initial_capital() -> float:
    """
    Retrieve account balance via MT5 to set INITIAL_CAPITAL.
    Falls back to a default value if MT5 is unavailable.

    Returns:
        float: Account balance in account currency.
    """
    try:
        if not mt5.initialize():
            logger.warning("MT5 initialization failed, using default capital.")
            return 10000.0

        account_info = mt5.account_info()
        if account_info is None:
            logger.warning("Cannot retrieve account info, using default capital.")
            return 10000.0

        balance: float = account_info.balance
        logger.info(f"Account balance (INITIAL_CAPITAL): {balance}")
        return balance

    except Exception as e:
        logger.error(f"Error retrieving account balance: {str(e)}")
        return 10000.0
    finally:
        mt5.shutdown()

INITIAL_CAPITAL: float = get_initial_capital()

# Dynamically calculate PIP_VALUE per symbol
def get_pip_value(symbol: str) -> float:
    """
    Retrieve pip value for a given symbol via MT5.
    Falls back to a default value if MT5 is unavailable.

    Args:
        symbol (str): The symbol (e.g., "GER40.cash").

    Returns:
        float: Pip value in account currency.
    """
    try:
        if not mt5.initialize():
            logger.warning(f"MT5 initialization failed for {symbol}, using default pip value.")
            return 10.0

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"Cannot retrieve symbol info for {symbol}, using default pip value.")
            return 10.0

        pip_value = symbol_info.point * symbol_info.trade_contract_size
        logger.info(f"Pip value for {symbol}: {pip_value}")
        return pip_value

    except Exception as e:
        logger.error(f"Error retrieving pip value for {symbol}: {str(e)}")
        return 10.0
    finally:
        mt5.shutdown()

PIP_VALUES: dict = {symbol: get_pip_value(symbol) for symbol in SYMBOLS}

OUTPUT_DIR: Path = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)