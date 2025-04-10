#!/bin/bash

TIMEFRAMES=("M5" "H1" "D1")
DAYS=90
STRATEGY="BollongStrategy"
SYMBOL="GER40.cash"

for TF in "${TIMEFRAMES[@]}"; do
    echo "Running backtest on $TF..."
    python main.py --mode backtest --strategy $STRATEGY --symbol $SYMBOL --timeframe $TF --days $DAYS --window 20 --std_dev 2.0 --sl_fixed_percent 0.015 --tp_fixed_percent 0.03 --risk_per_trade 0.01 --confidence_level 0.95
    echo "Running optimization on $TF..."
    python main.py --mode optimize --strategy $STRATEGY --symbol $SYMBOL --timeframe $TF --days $DAYS --metric sharpe_ratio --top_n 5 --quick
done