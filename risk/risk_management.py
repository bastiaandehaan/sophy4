# Sophy4/risk/risk_management.py
from typing import Union, List

import numpy as np
import pandas as pd

from config import logger


class RiskManager:
    def __init__(self, confidence_level: float = 0.95, holding_period: int = 1,
                 max_risk: float = 0.01, max_drawdown: float = 0.2):
        """
        Initialiseer de RiskManager met VaR-gebaseerde parameters.

        Args:
            confidence_level: Betrouwbaarheidsniveau voor VaR (bijv. 0.95 voor 95%).
            holding_period: Houdperiode in dagen voor VaR-berekening.
            max_risk: Maximale risico per transactie als fractie van kapitaal (bijv. 0.01 voor 1%).
            max_drawdown: Maximale toegestane drawdown als fractie (bijv. 0.2 voor 20%).

        Raises:
            ValueError: Als parameters ongeldig zijn.
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level moet tussen 0 en 1 liggen")
        if holding_period < 1:
            raise ValueError("holding_period moet een positieve integer zijn")
        if not 0 < max_risk <= 0.1:
            raise ValueError("max_risk moet tussen 0 en 0.1 liggen")
        if not 0 < max_drawdown <= 1:
            raise ValueError("max_drawdown moet tussen 0 en 1 liggen")

        self.confidence_level: float = confidence_level
        self.holding_period: int = holding_period
        self.max_risk: float = max_risk
        self.max_drawdown: float = max_drawdown

    def calculate_historical_var(self, returns: Union[pd.Series, np.ndarray, List[float]]) -> float:
        """
        Bereken historische VaR op basis van rendementen.

        Args:
            returns: Historische rendementen van het instrument.

        Returns:
            VaR-waarde (potentieel verlies, positief getal).
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        elif isinstance(returns, list):
            returns = np.array(returns)

        if len(returns) == 0:
            logger.warning("Geen rendementen beschikbaar voor VaR-berekening.")
            return 0.0

        var: float = np.percentile(returns, (1 - self.confidence_level) * 100)
        var_adjusted: float = -var * np.sqrt(self.holding_period)
        logger.info(f"VaR berekend: {var_adjusted:.4f} (confidence={self.confidence_level}, holding_period={self.holding_period})")
        return var_adjusted

    def calculate_position_size(self, capital: float, returns: Union[pd.Series, np.ndarray, List[float]],
                                pip_value: float) -> float:
        """
        Bereken positiegrootte op basis van VaR en risico per transactie.

        Args:
            capital: Huidige accountbalans/kapitaal.
            returns: Historische rendementen voor VaR-berekening.
            pip_value: Waarde per pip voor het instrument (bijv. 10.0 voor forexparen).

        Returns:
            Positiegrootte in units/lots.
        """
        if pip_value <= 0:
            raise ValueError("pip_value moet positief zijn")

        var: float = self.calculate_historical_var(returns)
        if var == 0:
            logger.error("VaR is 0, kan positiegrootte niet berekenen.")
            return 0.0

        risk_per_trade: float = capital * self.max_risk
        position_size: float = risk_per_trade / (abs(var) * pip_value)
        position_size = round(position_size, 2)

        logger.info(f"Position Size: {position_size:.2f} units (Capital={capital:.2f}, VaR={var:.4f}, Risk={self.max_risk * 100}%)")
        return position_size

    def monitor_drawdown(self, current_value: float, max_value: float) -> bool:
        """
        Controleer of de huidige drawdown de maximale limiet overschrijdt.

        Args:
            current_value: Huidige portefeuillewaarde.
            max_value: Maximale portefeuillewaarde tot nu toe.

        Returns:
            True als drawdown limiet overschreden is, anders False.
        """
        if max_value <= 0:
            logger.warning("Maximale waarde is 0 of negatief, drawdown kan niet berekend worden.")
            return False

        drawdown: float = (max_value - current_value) / max_value
        exceeds_limit: bool = drawdown > self.max_drawdown
        if exceeds_limit:
            logger.warning(f"Drawdown overschreden: {drawdown:.4f} > {self.max_drawdown}")
        else:
            logger.debug(f"Huidige drawdown: {drawdown:.4f}, binnen limiet {self.max_drawdown}")
        return exceeds_limit


def calculate_position_size(capital: float, price: float, sl_percent: float, max_risk: float = 0.01) -> float:
    """
    Oude functie voor positiegrootte gebaseerd op stop-loss percentage (voor compatibiliteit).

    Args:
        capital: Huidige accountbalans
        price: Huidige prijs van het instrument
        sl_percent: Stop-loss percentage
        max_risk: Maximale risico per transactie

    Returns:
        Positiegrootte in units
    """
    risk_per_trade: float = capital * max_risk
    position_size: float = risk_per_trade / (price * sl_percent)
    logger.info(f"Position Size (oude methode): {position_size:.2f} units (Risk={max_risk * 100}%)")
    return position_size


if __name__ == "__main__":
    rm: RiskManager = RiskManager(confidence_level=0.95, holding_period=1, max_risk=0.01, max_drawdown=0.2)
    sample_returns: pd.Series = pd.Series(np.random.normal(0, 0.01, 100))
    capital: float = 10000.0
    pip_value: float = 10.0

    size: float = rm.calculate_position_size(capital, sample_returns, pip_value)
    print(f"Positiegrootte: {size}")

    current_value: float = 9500.0
    max_value: float = 10500.0
    drawdown_exceeded: bool = rm.monitor_drawdown(current_value, max_value)
    print(f"Drawdown overschreden: {drawdown_exceeded}")