# Sophy4/config.py
import logging
from pathlib import Path

# Logging configuratie
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
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
