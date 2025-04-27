import logging
from typing import Dict, Optional, Tuple, Any

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management class with VectorBT integration."""
    def __init__(self, confidence_level: float = 0.95, max_risk: float = 0.01,
                 max_daily_loss_percent: float = 0.05, max_total_loss_percent: float = 0.10,
                 correlated_symbols: Optional[Dict[str, list]] = None):
        self.confidence_level = confidence_level
        self.max_risk = max_risk
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_total_loss_percent = max_total_loss_percent
        self.correlated_symbols = correlated_symbols or {}
        self.var_cache: Dict[Tuple, float] = {}

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, float]]:
        """Retrieve symbol information via MT5 API."""
        try:
            if not mt5.initialize():
                logger.warning(f"MT5 initialization failed for {symbol}.")
                return None

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Cannot retrieve symbol info for {symbol}.")
                return None

            pip_value = symbol_info.point * symbol_info.trade_contract_size
            spread = symbol_info.spread * symbol_info.point
            tick_value = symbol_info.trade_tick_value if hasattr(symbol_info, 'trade_tick_value') else 10.0

            return {"pip_value": pip_value, "spread": spread, "tick_value": tick_value,
                    "tick_size": symbol_info.tick_size, "contract_size": symbol_info.trade_contract_size}
        except Exception as e:
            logger.error(f"Error retrieving symbol info for {symbol}: {str(e)}")
            return None
        finally:
            mt5.shutdown()

    def is_market_open(self, symbol: str) -> bool:
        """Check if the market is open for the given symbol."""
        try:
            if not mt5.initialize():
                logger.warning(f"MT5 initialization failed for {symbol}.")
                return False

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Cannot retrieve symbol info for {symbol}.")
                return False

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick data for {symbol}.")
                return False

            spread = (tick.ask - tick.bid) / symbol_info.point
            return 0 < spread < 1000
        except Exception as e:
            logger.error(f"Error checking market status for {symbol}: {str(e)}")
            return False
        finally:
            mt5.shutdown()

    def calculate_var(self, returns: pd.Series, capital: float,
                      symbol: Optional[str] = None,
                      open_positions: Optional[Dict[str, int]] = None) -> float:
        """Calculate Value at Risk (VaR) using VectorBT."""
        if len(returns) < 10 or returns.empty:
            logger.warning("No valid returns data for VaR calculation, returning default VaR")
            return 0.01

        cache_key = (tuple(returns.values), capital, symbol if symbol else "",
                     tuple(sorted(open_positions.items())) if open_positions else ())

        if cache_key in self.var_cache:
            return self.var_cache[cache_key]

        # Use VectorBT for portfolio returns
        portfolio_returns = vbt.Returns.from_pandas(returns)
        if symbol and open_positions and self.correlated_symbols:
            correlated = self.correlated_symbols.get(symbol, [])
            total_positions = sum(open_positions.values())
            if total_positions > 0:
                weights = np.ones(len(correlated) + 1) / (len(correlated) + 1)
                portfolio_returns = portfolio_returns.values * weights[0]
                for i, corr_symbol in enumerate(correlated):
                    if corr_symbol in open_positions:
                        portfolio_returns += portfolio_returns.values * weights[i + 1]

        mean = np.mean(portfolio_returns)
        std_dev = np.std(portfolio_returns)
        z_score = norm.ppf(self.confidence_level)
        var = capital * (mean - z_score * std_dev)

        self.var_cache[cache_key] = var
        logger.debug(f"Calculated VaR for {symbol}: {var:.2f}")
        return var

    def calculate_position_size(self, capital: float, returns: pd.Series,
                                pip_value: float, symbol: Optional[str] = None,
                                open_positions: Optional[Dict[str, int]] = None) -> float:
        """Calculate position size using VectorBT."""
        effective_pip_value = pip_value
        spread_cost = 0.0
        if symbol:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                effective_pip_value = symbol_info["pip_value"]
                spread_cost = symbol_info["spread"]
            if not self.is_market_open(symbol):
                logger.warning(f"Market is closed for {symbol}. Position size set to 0.")
                return 0.0

        risk_amount = capital * self.max_risk
        var = self.calculate_var(returns, capital, symbol, open_positions)
        risk_amount = min(risk_amount, var)

        points_at_risk = capital * 0.01 / effective_pip_value
        points_at_risk = max(points_at_risk, 100)  # Default to 100 points

        total_risk_per_unit = (points_at_risk * effective_pip_value) + spread_cost
        position_size = risk_amount / total_risk_per_unit
        position_size = max(0.01, min(position_size, 10.0))  # Limit between 0.01 and 10 lots

        logger.info(f"Calculated position size for {symbol}: {position_size:.2f} lots")
        return position_size

    def calculate_adjusted_position_size(self, capital: float, returns: pd.Series,
                                         symbol: Optional[str] = None, price: Optional[float] = None,
                                         open_positions: Optional[Dict[str, int]] = None) -> float:
        """Advanced position size calculation."""
        pip_value = 10.0
        if symbol:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                pip_value = symbol_info["pip_value"]
            else:
                try:
                    from config import PIP_VALUES
                    pip_value = PIP_VALUES.get(symbol, 10.0)
                except (ImportError, KeyError):
                    logger.warning(f"No pip value found for {symbol}, using default: {pip_value}")

        sl_distance = getattr(self, 'sl_fixed_percent', 0.01)
        risk_amount = capital * self.max_risk
        var = self.calculate_var(returns, capital, symbol, open_positions)
        risk_amount = min(risk_amount, var)

        if price and price > 0:
            price_risk = price * sl_distance
            position_size = risk_amount / price_risk
        else:
            points_at_risk = capital * 0.01 / pip_value
            position_size = risk_amount / (points_at_risk * pip_value)

        position_size = max(0.01, min(position_size, 10.0))
        logger.info(f"Adjusted position size for {symbol}: {position_size:.2f} lots")
        return position_size

    def monitor_drawdown(self, current_capital: float, max_value: float) -> bool:
        """Monitor drawdown against FTMO limits."""
        daily_loss = (max_value - current_capital) / max_value
        total_loss = (max_value - current_capital) / max_value

        if daily_loss > self.max_daily_loss_percent:
            logger.error(f"Daily loss ({daily_loss:.2%}) exceeds FTMO limit ({self.max_daily_loss_percent:.2%})")
            return True
        if total_loss > self.max_total_loss_percent:
            logger.error(f"Total loss ({total_loss:.2%}) exceeds FTMO limit ({self.max_total_loss_percent:.2%})")
            return True
        return False

    def get_max_daily_loss(self, capital: float) -> float:
        """Calculate maximum allowable daily loss."""
        max_loss = capital * self.max_daily_loss_percent
        logger.debug(f"Max daily loss for capital {capital}: {max_loss:.2f}")
        return max_loss

    def get_max_total_loss(self, capital: float) -> float:
        """Calculate maximum allowable total loss."""
        max_loss = capital * self.max_total_loss_percent
        logger.debug(f"Max total loss for capital {capital}: {max_loss:.2f}")
        return max_loss

    def clear_cache(self) -> None:
        """Clear the VaR cache."""
        self.var_cache.clear()
        logger.info("VaR cache cleared")