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

## System Architecture

The Sophy4 framework is composed of multiple modular components:
- `backtest/`: Handles backtesting logic with VectorBT.
- `live/`: Placeholder for live trading execution (TBD).
- `risk/`: Implements VaR-based risk management.
- `monitor/`: Tracks performance metrics.
- `ftmo_compliance/`: Checks FTMO rules.
- `strategies/`: Defines trading strategies (e.g., BollongStrategy).
- `utils/`: Utility functions for data and indicators.

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
Located in `strategies/`, defines trading strategies. Example: `bollong.py` implements a Bollinger Bands breakout strategy with VaR-based sizing.

### Utilities Module
Located in `utils/`, provides helper functions (e.g., `calculate_bollinger_bands`).

## Developing Strategies

1. Create a new file in `strategies/` (e.g., `my_strategy.py`).
2. Inherit from `BaseStrategy`:
   ```python
   from strategies.base_strategy import BaseStrategy
   from typing import Tuple
   import pandas as pd

   class MyStrategy(BaseStrategy):
       def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
           # Implement signal logic
           pass