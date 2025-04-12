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

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run a backtest:
   ```
   python main.py --mode backtest --strategy BollongStrategy --symbol GER40.cash
   ```

3. Run live trading (when available):
   ```
   python main.py --mode live --strategy BollongStrategy --symbol GER40.cash
   ```

## Available Strategies

- **BollongStrategy**: A long-only Bollinger Bands breakout strategy.

## Configuration

Edit configuration settings in `config.py` including:
- Initial capital
- Risk parameters
- FTMO compliance settings

## Documentation

For more detailed information, please refer to the `user_manual.md` which contains comprehensive documentation about:
- System architecture
- Module descriptions
- Creating custom strategies
- Risk management features
- Performance monitoring

## Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:
- VectorBT 0.27.2
- Pandas 2.2.3
- NumPy 2.1.3
- MetaTrader5 5.0.4874