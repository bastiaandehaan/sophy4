# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

import pandas as pd
import MetaTrader5 as mt5


class BaseStrategy(ABC):
    """
    Abstracte basisklasse die alle trading strategieën moeten implementeren.
    Zorgt voor een uniforme interface voor backtest, optimalisatie en live trading.
    """

    def __init__(self) -> None:
        self.name: str = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, current_capital: Optional[float] = None) -> Tuple[pd.Series, ...]:
        """
        Genereer trading signalen op basis van de data.

        Args:
            df: DataFrame met OHLC data
            current_capital: Huidig kapitaal (optioneel, voor dynamische positiegrootte)

        Returns:
            Tuple van Series: Minimaal (entries, sl_stop, tp_stop), optioneel meer (bijv. trailing_stop, position_sizes)
            - entries: 1 voor kopen, -1 voor verkopen, 0 voor niets
            - sl_stop: Stop-loss percentage
            - tp_stop: Take-profit percentage
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

    @staticmethod
    def fetch_historical_data(symbol: str, timeframe: Union[str, int] = mt5.TIMEFRAME_D1,
                              days: int = 1095, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Haalt historische data op via MetaTrader 5.

        Deze methode is een doorverwijzing naar backtest.data_loader.fetch_historical_data
        en biedt strategieën direct toegang tot historische data.

        Args:
            symbol: Trading symbool (bijv. 'GER40.cash').
            timeframe: MT5 timeframe constante of string (bijv. mt5.TIMEFRAME_D1 of 'D1').
            days: Aantal dagen historische data.
            end_date: Einddatum (standaard: nu).

        Returns:
            DataFrame met OHLC-data of None bij fout.
        """
        # Importeer hier om circulaire imports te voorkomen
        from backtest.data_loader import fetch_historical_data as fetch_data
        return fetch_data(symbol, timeframe, days, end_date)

    @staticmethod
    def fetch_live_data(symbol: str, timeframe: Union[str, int] = mt5.TIMEFRAME_D1) -> Optional[pd.DataFrame]:
        """
        Haalt de meest recente candle op via MetaTrader 5.

        Deze methode is een doorverwijzing naar backtest.data_loader.fetch_live_data
        en biedt strategieën direct toegang tot live marktdata.

        Args:
            symbol: Trading symbool (bijv. 'GER40.cash').
            timeframe: MT5 timeframe constante of string (bijv. mt5.TIMEFRAME_D1 of 'D1').

        Returns:
            DataFrame met de laatste candle of None bij fout.
        """
        # Importeer hier om circulaire imports te voorkomen
        from backtest.data_loader import fetch_live_data as fetch_data
        return fetch_data(symbol, timeframe)