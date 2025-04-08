import logging
from pathlib import Path

# Logging configuratie met bestand
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "sophy4_backtest.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,  # Logs naar een bestand schrijven
    filemode='a'        # Append-modus (voegt toe aan bestaand bestand)
)
logger = logging.getLogger()

# FTMO limieten
MAX_DAILY_LOSS = 0.05  # 5%
MAX_TOTAL_LOSS = 0.10  # 10%
PROFIT_TARGET = 0.10  # 10%

# Standaardparameters
INITIAL_CAPITAL = 10000.0
SYMBOL = "GER40.cash"
FEES = 0.0002  # 0.02%
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Add this after your existing logger setup
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)