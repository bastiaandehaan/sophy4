import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from scipy.stats import norm

# Configureer een lokale logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk management class to ensure compliance with FTMO rules and manage portfolio risk.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        max_risk: float = 0.01,
        max_daily_loss_percent: float = 0.05,
        max_total_loss_percent: float = 0.10,
        correlated_symbols: Optional[Dict[str, list]] = None
    ):
        """
        Initialize the RiskManager with risk parameters.

        Args:
            confidence_level (float): Confidence level for VaR calculation.
            max_risk (float): Maximum risk per trade as a percentage of portfolio.
            max_daily_loss_percent (float): Maximum daily loss percentage (e.g., 0.05 for 5%).
            max_total_loss_percent (float): Maximum total loss percentage (e.g., 0.10 for 10%).
            correlated_symbols (Dict[str, list], optional): Dictionary of correlated symbols.
        """
        self.confidence_level = confidence_level
        self.max_risk = max_risk
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_total_loss_percent = max_total_loss_percent
        self.correlated_symbols = correlated_symbols or {}
        self.var_cache: Dict[Tuple, float] = {}

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Retrieve symbol information via MT5 API.

        Args:
            symbol (str): The symbol (e.g., "US30.cash").

        Returns:
            Dict[str, float]: Dictionary with symbol information (pip_value, spread, etc.), or None if failed.
        """
        try:
            if not mt5.initialize():
                logger.warning(f"MT5 initialization failed for {symbol}.")
                return None

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Cannot retrieve symbol info for {symbol}.")
                return None

            # Use trade_tick_value for tick value
            tick_value = symbol_info.trade_tick_value if hasattr(symbol_info, 'trade_tick_value') else 10.0

            # Bereken pip-waarde dynamisch
            pip_value = symbol_info.point * symbol_info.trade_contract_size
            spread = symbol_info.spread * symbol_info.point  # Spread in prijs-eenheden

            return {
                "pip_value": pip_value,
                "spread": spread,
                "tick_value": tick_value,
                "tick_size": symbol_info.tick_size,
                "contract_size": symbol_info.trade_contract_size
            }

        except Exception as e:
            logger.error(f"Error retrieving symbol info for {symbol}: {str(e)}")
            return None
        finally:
            mt5.shutdown()

    def is_market_open(self, symbol: str) -> bool:
        """
        Check if the market is open for the given symbol.

        Args:
            symbol (str): The symbol (e.g., "US30.cash").

        Returns:
            bool: True if market is open, False otherwise.
        """
        try:
            if not mt5.initialize():
                logger.warning(f"MT5 initialization failed for {symbol}. Assuming market is closed.")
                return False

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Cannot retrieve symbol info for {symbol}. Assuming market is closed.")
                return False

            # Controleer of er recente ticks zijn (indicatie dat markt open is)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick data for {symbol}. Assuming market is closed.")
                return False

            # Als de bid/ask prijs beschikbaar is en de spread niet extreem groot is, is de markt waarschijnlijk open
            spread = (tick.ask - tick.bid) / symbol_info.point
            if spread > 0 and spread < 1000:  # Arbitraire grens voor "redelijke" spread
                return True
            return False

        except Exception as e:
            logger.error(f"Error checking market status for {symbol}: {str(e)}")
            return False
        finally:
            mt5.shutdown()

    def calculate_var(
        self,
        returns: pd.Series,
        capital: float,
        symbol: Optional[str] = None,
        open_positions: Optional[Dict[str, int]] = None
    ) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.

        Args:
            returns (pd.Series): Historical returns of the asset.
            capital (float): Current portfolio capital.
            symbol (str, optional): Symbol to calculate VaR for.
            open_positions (Dict[str, int], optional): Dictionary of open positions.

        Returns:
            float: VaR value at the specified confidence level.
        """
        if len(returns) < 10 or returns.empty:
            logger.warning("No valid returns data provided for VaR calculation, returning default VaR")
            return 0.01  # Default VaR

        # Create a cache key based on inputs
        cache_key = (
            tuple(returns.values),
            capital,
            symbol if symbol else "",
            tuple(sorted(open_positions.items())) if open_positions else ()
        )

        if cache_key in self.var_cache:
            return self.var_cache[cache_key]

        # Calculate portfolio returns considering correlations
        portfolio_returns = returns.values
        if symbol and open_positions and self.correlated_symbols:
            correlated = self.correlated_symbols.get(symbol, [])
            total_positions = sum(open_positions.values())
            if total_positions > 0:
                weights = np.ones(len(correlated) + 1) / (len(correlated) + 1)
                portfolio_returns = returns.values * weights[0]
                for i, corr_symbol in enumerate(correlated):
                    if corr_symbol in open_positions:
                        portfolio_returns += returns.values * weights[i + 1]

        # Calculate VaR using historical simulation
        mean = np.mean(portfolio_returns)
        std_dev = np.std(portfolio_returns)
        z_score = norm.ppf(self.confidence_level)
        var = capital * (mean - z_score * std_dev)

        self.var_cache[cache_key] = var
        logger.debug(f"Calculated VaR for {symbol}: {var:.2f}")
        return var

    def calculate_position_size(
        self,
        capital: float,
        returns: pd.Series,
        pip_value: float,
        symbol: Optional[str] = None,
        open_positions: Optional[Dict[str, int]] = None
    ) -> float:
        """
        Calculate the position size based on risk parameters, VaR, and dynamic market data.

        Args:
            capital (float): Current portfolio capital.
            returns (pd.Series): Historical returns of the asset.
            pip_value (float): Pip value for the symbol (can be overridden by MT5 data if symbol is provided).
            symbol (str, optional): Symbol to trade (for VaR correlation and dynamic data).
            open_positions (Dict[str, int], optional): Dictionary of open positions.

        Returns:
            float: Position size.
        """
        # Haal dynamische gegevens op als symbol is meegegeven
        effective_pip_value = pip_value
        spread_cost = 0.0
        if symbol:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                effective_pip_value = symbol_info["pip_value"]
                spread_cost = symbol_info["spread"]
                logger.info(f"Using dynamic pip value for {symbol}: {effective_pip_value}, spread: {spread_cost}")
            else:
                logger.warning(f"Failed to retrieve symbol info for {symbol}, using provided pip value: {pip_value}")

            # Controleer marktstatus (alleen relevant voor live trading, optioneel in backtest)
            if not self.is_market_open(symbol):
                logger.warning(f"Market is closed for {symbol}. Position size set to 0.")
                return 0.0

        risk_amount = capital * self.max_risk
        var = self.calculate_var(returns, capital, symbol, open_positions)

        # Pas risico aan op basis van VaR
        if var > risk_amount:
            logger.warning(f"VaR ({var:.2f}) exceeds risk limit ({risk_amount:.2f}). Reducing position size.")
            risk_amount = min(risk_amount, var)

        # Schat punten in gevaar (gebruik een 1% prijsbeweging als proxy, of pas aan met sl_fixed_percent als beschikbaar)
        points_at_risk = capital * 0.01 / effective_pip_value  # 1% van kapitaal als risicopunten
        if points_at_risk <= 0:
            logger.warning("Points at risk is zero, using default value")
            points_at_risk = 100  # Default naar 100 punten

        # Voeg spread-kosten toe aan het risico
        total_risk_per_unit = (points_at_risk * effective_pip_value) + spread_cost
        if total_risk_per_unit <= 0:
            logger.warning("Total risk per unit is zero or negative, using default")
            total_risk_per_unit = points_at_risk * effective_pip_value

        # Bereken positiegrootte: risk_amount = position_size * total_risk_per_unit
        position_size = risk_amount / total_risk_per_unit
        if position_size <= 0:
            logger.warning("Calculated position size is zero or negative, using minimum size")
            position_size = 0.01  # Minimale positiegrootte

        # Beperk tot een redelijke maximum grootte
        position_size = min(position_size, 10.0)  # Max 10 lots

        logger.info(f"Calculated position size for {symbol}: {position_size:.2f} lots (including spread cost: {spread_cost})")
        return position_size

    def monitor_drawdown(self, current_capital: float, max_value: float) -> bool:
        """
        Monitor drawdown against FTMO limits.

        Args:
            current_capital (float): Current portfolio capital.
            max_value (float): Maximum portfolio value achieved.

        Returns:
            bool: True if drawdown exceeds limits, False otherwise.
        """
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
        """
        Calculate the maximum allowable daily loss in currency units.

        Args:
            capital (float): Current portfolio capital.

        Returns:
            float: Maximum daily loss in currency units.
        """
        max_loss = capital * self.max_daily_loss_percent
        logger.debug(f"Max daily loss for capital {capital}: {max_loss:.2f}")
        return max_loss

    def get_max_total_loss(self, capital: float) -> float:
        """
        Calculate the maximum allowable total loss in currency units.

        Args:
            capital (float): Current portfolio capital.

        Returns:
            float: Maximum total loss in currency units.
        """
        max_loss = capital * self.max_total_loss_percent
        logger.debug(f"Max total loss for capital {capital}: {max_loss:.2f}")
        return max_loss

    def clear_cache(self) -> None:
        """
        Clear the VaR cache to free memory.
        """
        self.var_cache.clear()
        logger.info("VaR cache cleared")