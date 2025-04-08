# strategies/base_strategy.py
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstracte basisklasse die alle trading strategieën moeten implementeren.
    Zorgt voor een uniforme interface voor backtest, optimalisatie en live trading.
    """
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df):
        """
        Genereer trading signalen op basis van de data.
        Args:
            df: DataFrame met OHLC data
        Returns:
            tuple: (entries, sl_stop, tp_stop) - Series met entry signalen en stop percentages
        """
        pass

    def validate_parameters(self):
        """
        Controleer of alle parameters geldig zijn.
        Gooit een exception als parameters ongeldig zijn.
        """
        return True

    @classmethod
    def get_default_params(cls):
        """
        Geeft default parameters voor de strategie terug.
        Wordt gebruikt voor optimalisatie.
        Returns:
            dict: Parameter namen en waarden voor grid search
        """
        return {}

    @classmethod
    def get_parameter_descriptions(cls):
        """
        Beschrijft wat elke parameter doet voor documentatie.
        Returns:
            dict: Parameter namen en beschrijvingen
        """
        return {}

    @classmethod
    def get_performance_metrics(cls):
        """
        Definieert welke performance metrics belangrijk zijn voor deze strategie.
        Wordt gebruikt door optimizer om te sorteren.
        Returns:
            list: Metrics in volgorde van belangrijkheid
        """
        return ["sharpe_ratio", "calmar_ratio", "total_return", "max_drawdown", "win_rate"]