Sophy4 Trading Framework
A modular trading framework for backtesting, live trading, risk management, monitoring, and FTMO compliance.
Structure

backtest/: Backtesting logic
live/: Live trading execution
risk/: Risk management tools
monitor/: Performance monitoring
ftmo_compliance/: FTMO rule checker
strategies/: Trading strategies (e.g., Bollong)
utils/: Helper functions
results/: Output directory

Installation

Install dependencies:
pip install -e .


Run a backtest:
python main.py backtest run BollongStrategy --symbol GER40.cash --timeframe H1 --days 1095


Run live trading (when available):
python main.py monitor live --symbols GER40.cash



Available Strategies

BollongStrategy: A long-only Bollinger Bands breakout strategy.

Configuration
Configuration settings can be adjusted in config.py, including:

Initial capital
Risk management parameters
FTMO compliance settings
Timeframe settings in timeframe_config.json

Documentation
For more detailed information, please refer to the user_manual.md which contains comprehensive documentation about:

System architecture
Module descriptions
Creating custom strategies
Risk management features
Performance monitoring

Requirements
See pyproject.toml for a complete list of dependencies. Key packages include:

VectorBT 0.27.2+
Pandas 2.2.3+
NumPy 2.1.3+
MetaTrader5 5.0.4874+

Contributing
Contributions are welcome! Feel free to submit a pull request or report issues.
