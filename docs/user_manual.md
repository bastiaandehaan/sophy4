# Sophy4 Trading Framework Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Modules](#modules)
    - [Backtest Module](#backtest-module)
    - [Live Trading Module](#live-trading-module)
    - [Risk Management Module](#risk-management-module)
    - [Monitoring Module](#monitoring-module)
    - [FTMO Compliance Module](#ftmo-compliance-module)
    - [Strategies Module](#strategies-module)
    - [Utilities Module](#utilities-module)
    - [Optimization Module](#optimization-module)
6. [Developing Strategies](#developing-strategies)
7. [Commands and Parameters](#commands-and-parameters)
8. [Performance Monitoring and Reporting](#performance-monitoring-and-reporting)
9. [Logging](#logging)
10. [Frequently Asked Questions](#frequently-asked-questions)

## Introduction

Sophy4 is a modular trading framework designed to support backtesting, live trading, risk management, monitoring, and FTMO compliance. Built with scalability and extensibility, it allows users to implement and test custom trading strategies using Python and VectorBT.

The system offers:
- **Comprehensive backtesting**: Test strategies on historical data with realistic execution simulation.
- **Live trading capability**: Execute strategies in real-time markets (module in development).
- **Risk management tools**: Calculate position sizes using Value at Risk (VaR) methodology.
- **Performance monitoring**: Track and visualize key performance metrics.
- **FTMO compliance checks**: Ensure strategies meet FTMO funding requirements.
- **Strategy optimization**: Optimize strategy parameters for maximum performance.

## System Architecture

The Sophy4 framework is composed of multiple modular components:
- `backtest/`: Handles backtesting logic with VectorBT.
- `live/`: Placeholder for live trading execution (TBD).
- `risk/`: Implements VaR-based risk management.
- `monitor/`: Tracks performance metrics.
- `ftmo_compliance/`: Checks FTMO rules.
- `strategies/`: Defines trading strategies (e.g., BollongStrategy).
- `utils/`: Utility functions for data and indicators.
- `optimization/`: Strategy parameter optimization capabilities.
- `models/`: Machine learning model definitions and storage.

## Installation

1. Clone the repository: `git clone <repo_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run a backtest: `python main.py --mode backtest --strategy BollongStrategy --symbol GER40.cash`

Requirements:
- Python 3.8+
- VectorBT
- NumPy, Pandas, Matplotlib

## Configuration

Edit `config.py` to set:
- `INITIAL_CAPITAL`: Starting capital (default: 10000.0)
- `SYMBOL`: Default trading symbol (default: "GER40.cash")
- `FEES`: Transaction fees (default: 0.02%)
- `PIP_VALUE`: Value per pip (default: 10.0)
- Logging paths and FTMO limits

Additionally, `timeframe_config.json` contains timeframe-specific settings that strategies can use for different timeframe configurations.

## Modules

### Backtest Module
Located in `backtest/backtest.py`, this module uses VectorBT to run backtests, calculate metrics, and generate visualizations. It supports Monte Carlo and walk-forward testing.

### Live Trading Module
(TBD) Will handle real-time trade execution.

### Risk Management Module
Located in `risk/risk_management.py`, this module implements a `RiskManager` class using historical VaR to:
- Calculate position sizes based on capital, historical returns, and pip value.
- Monitor drawdown against a maximum limit.
- Key parameters: `confidence_level` (default: 0.95), `max_risk` (default: 0.01).

### Monitoring Module
Located in `monitor/monitor.py`, tracks performance metrics during backtesting or live trading.

### FTMO Compliance Module
Located in `ftmo_compliance/ftmo_check.py`, ensures strategies meet FTMO rules (e.g., max 10% drawdown).

### Strategies Module
Located in `strategies/`, defines trading strategies:
- `bollong.py`: Implements a Bollinger Bands breakout strategy with VaR-based sizing
- `bollong_vectorized.py`: Vectorized implementation of the Bollinger strategy
- `order_block_lstm_strategy.py`: Advanced strategy combining order blocks with LSTM predictions

### Utilities Module
Located in `utils/`, provides helper functions (e.g., `calculate_bollinger_bands`).

### Optimization Module
Located in `optimization/`, contains tools for parameter optimization to find the best-performing strategy settings.

## Developing Strategies

1. Create a new file in `strategies/` (e.g., `my_strategy.py`).
2. Inherit from `BaseStrategy`:
   ```python
   from strategies.base_strategy import BaseStrategy
   from typing import Tuple, Dict, List, Any, Optional
   import pandas as pd
   import json
   from utils.indicators import calculate_bollinger_bands, calculate_atr, calculate_adx
   from risk.risk_management import RiskManager
   import logging

   logger = logging.getLogger(__name__)

   # Use the register_strategy decorator to make it available to the framework
   from strategies import register_strategy

   @register_strategy
   class MyStrategy(BaseStrategy):
       def __init__(self, symbol: str = "EURUSD", window: int = 50, # other parameters):
           super().__init__()
           self.symbol = symbol
           self.window = window
           # Initialize other parameters
       
       def validate_parameters(self) -> bool:
           # Implement parameter validation
           return True
           
       def generate_signals(self, df: pd.DataFrame, current_capital: Optional[float] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
           # Implement signal logic
           # Return (entries, sl_stop, tp_stop)
           
       @classmethod
       def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
           # Return default parameters for optimization
           
       @classmethod
       def get_parameter_descriptions(cls) -> Dict[str, str]:
           # Return parameter descriptions
   ```

3. Implement the required methods:
   - `generate_signals()`: Core strategy logic that returns entry signals and stop levels
   - `validate_parameters()`: Validate strategy parameters
   - `get_default_params()`: Default parameters for optimization (optional)
   - `get_parameter_descriptions()`: Parameter descriptions for documentation (optional)

4. Use the strategy:
   ```
   python main.py --mode backtest --strategy MyStrategy --symbol EURUSD
   ```

## Commands and Parameters

### Main Script Options