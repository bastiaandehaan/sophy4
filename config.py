import logging
from pathlib import Path

import MetaTrader5 as mt5

# Logging configuration
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "sophy4_backtest.log"

# Basisconfig op ERROR niveau
logging.basicConfig(
    level=logging.ERROR,  # Alleen kritieke meldingen tonen
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,
    filemode='a'
)
logger = logging.getLogger()

# Verwijder eventuele bestaande handlers voordat we nieuwe toevoegen
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Voeg file handler toe op ERROR niveau
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Voeg console handler toe op ERROR niveau
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # Gebruik ERROR in plaats van INFO
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))  # Vereenvoudigde formatting
logger.addHandler(console_handler)

# FTMO limits
MAX_DAILY_LOSS = 0.05
MAX_TOTAL_LOSS = 0.10
PROFIT_TARGET = 0.10

# Trading symbols
SYMBOLS = ["EURUSD", "GER40.cash", "US30.cash", "XAUUSD"]

# Correlated symbols for risk management
CORRELATED_SYMBOLS = {
    'EURUSD': ['GBPUSD', 'USDCHF', 'EURGBP'],
    'XAUUSD': ['XAGUSD', 'USDCAD'],
    'GER40.cash': ['FRA40.cash', 'UK100.cash'],
    'US30.cash': ['US500.cash', 'NAS100.cash']
}

# Default parameters
FEES = 0.00005  # Spread-based (0.005%)

def get_initial_capital() -> float:
    """Retrieve account balance via MT5."""
    try:
        if not mt5.initialize():
            return 10000.0

        account_info = mt5.account_info()
        if account_info is None:
            return 10000.0

        balance = account_info.balance
        return balance
    except Exception:
        return 10000.0
    finally:
        mt5.shutdown()

INITIAL_CAPITAL = get_initial_capital()

def get_pip_value(symbol: str) -> float:
    """Retrieve pip value for a given symbol via MT5."""
    try:
        if not mt5.initialize():
            return 10.0

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 10.0

        pip_value = symbol_info.point * symbol_info.trade_contract_size
        return pip_value
    except Exception:
        return 10.0
    finally:
        mt5.shutdown()

PIP_VALUES = {symbol: get_pip_value(symbol) for symbol in SYMBOLS}

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# VectorBT specific configurations
VBT_SETTINGS = {
    'portfolio': {
        'init_cash': INITIAL_CAPITAL,
        'fees': FEES,
        'slippage': 0.0001,
        'size_granularity': 0.01
    },
    'optimization': {
        'n_trials': 100,
        'metric': 'sharpe_ratio'
    }
}

# TensorFlow specific configurations
TF_SETTINGS = {
    'lstm': {
        'seq_len': 50,
        'batch_size': 32,
        'epochs': 50
    }
}