# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstracte basisklasse die alle trading strategieën moeten implementeren.
    Zorgt voor een uniforme interface voor backtest, optimalisatie en live trading.
    """

    def __init__(self) -> None:
        self.name: str = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Genereer trading signalen op basis van de data.

        Args:
            df: DataFrame met OHLC data

        Returns:
            Tuple van (entries, sl_stop, tp_stop) - Series met entry signalen en stop percentages
        """
        pass

    def validate_parameters(self) -> bool:
        """
        Controleer of alle parameters geldig zijn.

        Returns:
            True als parameters geldig zijn, anders raise exception
        """
        return True

    @classmethod
    def get_default_params(cls) -> Dict[str, List[Any]]:
        """
        Geeft default parameters voor de strategie terug.

        Returns:
            Dictionary met parameter namen en waarden voor grid search
        """
        return {'risk_per_trade': [0.01], 'confidence_level': [0.95]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """
        Beschrijft wat elke parameter doet voor documentatie.

        Returns:
            Dictionary met parameter namen en beschrijvingen
        """
        return {
            'risk_per_trade': 'Risico per trade als percentage van portfolio',
            'confidence_level': 'Betrouwbaarheidsniveau voor VaR-berekening'
        }

    @classmethod
    def get_performance_metrics(cls) -> List[str]:
        """
        Definieert welke performance metrics belangrijk zijn voor deze strategie.

        Returns:
            Lijst met metrics in volgorde van belangrijkheid
        """
        return ["sharpe_ratio", "calmar_ratio", "total_return", "max_drawdown", "win_rate"]