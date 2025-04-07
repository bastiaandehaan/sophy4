# Sophy4 Trading Framework

A modular trading framework for backtesting, live trading, risk management, monitoring, and FTMO compliance.

## Structure
- **backtest/**: Backtesting logic
- **live/**: Live trading execution
- **risk/**: Risk management tools
- **monitor/**: Performance monitoring
- **ftmo_compliance/**: FTMO rule checker
- **strategies/**: Trading strategies (e.g., bollong)
- **utils/**: Helper functions
- **results/**: Output directory

## Installation
1. Install dependencies: pip install -r requirements.txt`n2. Run backtest: python main.py`n3. Switch to live mode: Edit main.py to main(mode='live')`n
## Strategy
- **Bollong**: A long-only Bollinger Bands breakout strategy.
