# Sophy4/backtest/backtest.py
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

from config import INITIAL_CAPITAL, FEES, logger, OUTPUT_DIR


def run_backtest(df: pd.DataFrame, symbol: str, strategy_params: Dict = None) -> Tuple[vbt.Portfolio, Dict]:
    """
    Voer een backtest uit met behulp van VectorBT op basis van signalen uit een strategie.

    Args:
        df: DataFrame met OHLC-data en signalen ('close', 'entries', 'sl_stop', 'tp_stop', optioneel 'sl_trail')
        symbol: Ticker van het instrument
        strategy_params: Optionele dictionary met strategieparameters (bijv. risk_per_trade, max_positions)

    Returns:
        Tuple van (Portfolio-object, dictionary met prestatiemetrics)
    """
    try:
        # Controleer vereiste kolommen
        required_cols = ['close', 'entries', 'sl_stop', 'tp_stop']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame mist vereiste kolommen: {required_cols}")

        # Stel positiegrootte in op basis van risk_per_trade (indien meegegeven)
        size = None
        if strategy_params and 'risk_per_trade' in strategy_params:
            risk_per_trade = strategy_params['risk_per_trade']
            # Gebruik inverse van sl_stop als proxy voor positiegrootte
            size = risk_per_trade / df['sl_stop'].replace(0, np.inf).clip(lower=0.01)
            if strategy_params.get('max_positions', float('inf')) < len(df['entries']):
                logger.warning(f"Aantal signalen overschrijdt max_positions ({strategy_params.get('max_positions')})")

        # Voer backtest uit
        portfolio_kwargs = {
            'close': df['close'],
            'entries': df['entries'],
            'sl_stop': df['sl_stop'],
            'tp_stop': df['tp_stop'],
            'init_cash': INITIAL_CAPITAL,
            'fees': FEES,
            'freq': '1D'
        }

        # Indien size parameter gebruikt wordt:
        if size is not None:
            portfolio_kwargs['size'] = size

        # NIET direct toevoegen: sl_trail=...
        pf = vbt.Portfolio.from_signals(**portfolio_kwargs)

        # Bereken metrics
        metrics = {
            'total_return': pf.total_return(),
            'sharpe_ratio': pf.sharpe_ratio(),
            'calmar_ratio': pf.calmar_ratio(),
            'sortino_ratio': pf.sortino_ratio(),
            'max_drawdown': pf.max_drawdown(),
            'win_rate': pf.trades.win_rate() if len(pf.trades) > 0 else 0,
            'trades_count': len(pf.trades)
        }

        # Log resultaten
        logger.info(
            f"Backtest Resultaten voor {symbol}: Return={metrics['total_return']:.2%}, "
            f"Sharpe={metrics['sharpe_ratio']:.2f}, Calmar={metrics['calmar_ratio']:.2f}, "
            f"Sortino={metrics['sortino_ratio']:.2f}, Max Drawdown={metrics['max_drawdown']:.2%}, "
            f"Win Rate={metrics['win_rate']:.2%}, Trades={metrics['trades_count']}"
        )

        # Plot equity curve
        plt.figure(figsize=(12, 6))
        pf.value().plot()
        plt.title(f"{symbol} Equity Curve")
        plt.savefig(OUTPUT_DIR / f"{symbol}_equity.png")
        plt.close()

        return pf, metrics

    except Exception as e:
        logger.error(f"Backtest mislukt voor {symbol}: {str(e)}")
        return None, {}