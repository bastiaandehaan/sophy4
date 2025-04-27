# strategies/bollong_vectorized.py
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from risk.risk_management import RiskManager
from strategies import register_strategy
from strategies.base_strategy import BaseStrategy

# Logger configureren
logger = logging.getLogger(__name__)


@register_strategy
class BollongVectorizedStrategy(BaseStrategy):
    """
    Gevectoriseerde versie van de BollongStrategy zonder afhankelijkheid
    van specifieke VectorBT indicator-modules.
    """

    def __init__(self, symbol: str = "EURUSD", window: int = 50, std_dev: float = 2.0,
                 sl_fixed_percent: float = 0.02, tp_fixed_percent: float = 0.03,
                 use_trailing_stop: bool = True, trailing_stop_percent: float = 0.015,
                 trailing_activation_percent: float = 0.01,
                 use_volume_filter: bool = False, volume_filter_periods: int = 20,
                 volume_filter_mult: float = 1.5, risk_per_trade: float = 0.01,
                 max_positions: int = 3, use_time_filter: bool = False,
                 trading_hours: Tuple[int, int] = (9, 17), min_adx: float = 20,
                 use_adx_filter: bool = False, confidence_level: float = 0.95):
        """
        Initialiseer de gevectoriseerde BollongStrategy.
        """
        super().__init__()
        self.symbol: str = symbol
        self.window: int = window
        self.std_dev: float = std_dev
        self.sl_fixed_percent: float = sl_fixed_percent
        self.tp_fixed_percent: float = tp_fixed_percent
        self.use_trailing_stop: bool = use_trailing_stop
        self.trailing_stop_percent: float = trailing_stop_percent
        self.trailing_activation_percent: float = trailing_activation_percent
        self.use_volume_filter: bool = use_volume_filter
        self.volume_filter_periods: int = volume_filter_periods
        self.volume_filter_mult: float = volume_filter_mult
        self.risk_per_trade: float = risk_per_trade
        self.max_positions: int = max_positions
        self.use_time_filter: bool = use_time_filter
        self.trading_hours: Tuple[int, int] = trading_hours
        self.min_adx: float = min_adx
        self.use_adx_filter: bool = use_adx_filter
        self.confidence_level: float = confidence_level
        self.sl_trail = False

    def validate_parameters(self) -> bool:
        """Controleer of alle parameters geldig zijn."""
        if self.window < 5:
            raise ValueError("Window moet ten minste 5 zijn")
        if self.std_dev <= 0:
            raise ValueError("std_dev moet positief zijn")
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            raise ValueError("risk_per_trade moet tussen 0 en 0.1 (10%) liggen")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level moet tussen 0 en 1 liggen")
        return True

    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Vectorized Bollinger Bands berekening.

        Args:
            prices: Series met prijsdata

        Returns:
            Tuple van (upper_band, middle_band, lower_band)
        """
        # Bereken SMA
        middle_band = prices.rolling(window=self.window).mean()

        # Bereken standaarddeviatie
        rolling_std = prices.rolling(window=self.window).std()

        # Bereken bands
        upper_band = middle_band + (rolling_std * self.std_dev)
        lower_band = middle_band - (rolling_std * self.std_dev)

        return upper_band, middle_band, lower_band

    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Vectorized ATR berekening.

        Args:
            df: DataFrame met high, low, close
            window: Perioden voor ATR

        Returns:
            ATR series
        """
        # Bereken True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr = tr.fillna(tr1)

        # Bereken ATR
        atr = tr.rolling(window=window).mean()
        atr = atr.fillna(tr)

        return atr

    def calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Vectorized ADX berekening.

        Args:
            df: DataFrame met high, low, close
            window: Perioden voor ADX

        Returns:
            ADX series
        """
        # Bereken True Range
        tr = self.calculate_atr(df, window)

        # Bereken +DM en -DM
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()

        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)

        # Bereken +DM
        condition1 = (high_diff > 0) & (high_diff > low_diff.abs())
        plus_dm = plus_dm.mask(condition1, high_diff)

        # Bereken -DM
        condition2 = (low_diff < 0) & (low_diff.abs() > high_diff)
        minus_dm = minus_dm.mask(condition2, low_diff.abs())

        # Bereken +DI en -DI
        tr_ma = tr.rolling(window=window).mean()
        plus_di = 100 * plus_dm.rolling(window=window).mean() / tr_ma
        minus_di = 100 * minus_dm.rolling(window=window).mean() / tr_ma

        # Bereken DX en ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,
                                                                              np.nan))
        adx = dx.rolling(window=window).mean().fillna(0)

        return adx

    def generate_signals(self, df: pd.DataFrame,
                         current_capital: Optional[float] = None) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Genereer trading signalen met gevectoriseerde berekeningen.

        Args:
            df: DataFrame met OHLC data
            current_capital: Huidig kapitaal (optioneel)

        Returns:
            Tuple van (entries, sl_stop, tp_stop)
        """
        logger.info(
            f"Genereren trading signalen voor {self.symbol} met gevectoriseerde BollongStrategy")
        has_datetime_index: bool = hasattr(df.index, 'hour')

        # --- Risk Management setup ---
        risk_manager: RiskManager = RiskManager(confidence_level=self.confidence_level,
            max_risk=self.risk_per_trade)
        returns: pd.Series = df['close'].pct_change().dropna()
        from config import INITIAL_CAPITAL, get_pip_value
        capital = current_capital or INITIAL_CAPITAL

        # Bereken positiegrootte
        try:
            pip_value = get_pip_value(self.symbol)
            logger.info(f"Pip value for {self.symbol}: {pip_value}")
        except Exception as e:
            pip_value = 10.0
            logger.warning(
                f"Kon pip value niet ophalen: {str(e)}, gebruik default: {pip_value}")

        size: float = risk_manager.calculate_position_size(capital, returns,
            pip_value=pip_value, symbol=self.symbol)
        logger.info(
            f"Calculated position size for {self.symbol}: {size:.2f} lots (including spread cost: 0.0)")

        # --- Vectoriseerde indicatoren ---
        # 1. Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(
            df['close'])

        # 2. ATR
        atr = self.calculate_atr(df)

        # 3. SMA voor trend filter
        sma = df['close'].rolling(window=100).mean()

        # --- Vectoriseerde signaal generatie ---
        # 1. Band width en relatieve price position
        band_width = upper_band - lower_band
        price_position = (df['close'] - lower_band) / band_width

        # 2. Entry condities
        entries = price_position > 0.7  # Boolean Series

        # 3. Volatiliteitsfilter
        avg_atr = atr.rolling(window=20).mean()
        volatility_filter = atr < avg_atr * 3.0

        # 4. Bull market filter
        bull_market = df['close'] > sma

        # 5. Combineer filters
        entries = entries & volatility_filter & bull_market

        # --- Extra filters ---
        # Volume filter
        if self.use_volume_filter and 'volume' in df.columns:
            avg_volume = df['volume'].rolling(window=self.volume_filter_periods).mean()
            volume_filter = df['volume'] > avg_volume * self.volume_filter_mult
            entries = entries & volume_filter

        # ADX filter
        if self.use_adx_filter:
            adx = self.calculate_adx(df)
            adx_filter = adx > self.min_adx
            entries = entries & adx_filter

        # Time filter
        if self.use_time_filter:
            if not has_datetime_index:
                raise ValueError("Time filter vereist een datetime-index")
            time_filter = pd.Series(df.index.hour.isin(
                range(self.trading_hours[0], self.trading_hours[1] + 1)),
                index=df.index)
            entries = entries & time_filter

        # --- Stop-loss en Take-profit ---
        sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
        tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

        # Trailing-stop logica
        if self.use_trailing_stop:
            self.sl_trail = True
            if self.trailing_activation_percent > 0:
                logger.info(
                    f"Trailing stop activeert na {self.trailing_activation_percent:.2%} prijsstijging")

        # Vul ontbrekende waarden
        sl_stop = sl_stop.fillna(0.015)
        tp_stop = tp_stop.fillna(0.09)

        # Integer Series voor entries
        entries = entries.astype(int)

        logger.info(f"Aantal LONG signalen: {entries.sum()}")
        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """Geef default parameters terug gebaseerd op timeframe."""
        try:
            with open("timeframe_config.json", "r") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(
                "Kan timeframe_config.json niet laden, gebruik fallback waarden")
            config = {
                "H1": {"window_range": [20, 50, 80], "std_dev_range": [1.5, 2.0, 2.5],
                    "sl_fixed_percent_range": [0.01, 0.02, 0.03],
                    "tp_fixed_percent_range": [0.02, 0.03, 0.04]}}

        tf_config = config.get(timeframe, config.get("H1", {}))

        return {'symbol': ["EURUSD", "GBPUSD", "XAUUSD"],
            'window': tf_config.get("window_range", [20, 50, 80]),
            'std_dev': tf_config.get("std_dev_range", [1.5, 2.0, 2.5]),
            'sl_fixed_percent': tf_config.get("sl_fixed_percent_range",
                                              [0.01, 0.02, 0.03]),
            'tp_fixed_percent': tf_config.get("tp_fixed_percent_range",
                                              [0.02, 0.03, 0.04]),
            'use_trailing_stop': [True], 'trailing_stop_percent': [0.01, 0.015, 0.02],
            'risk_per_trade': [0.005, 0.01], 'confidence_level': [0.90, 0.95, 0.99]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """Beschrijf parameters."""
        descriptions = {
            'symbol': 'Handelssymbool voor de strategie (bijv. EURUSD, GBPUSD, XAUUSD)',
            'window': 'Aantal perioden voor Bollinger Bands (voortschrijdend gemiddelde)',
            'std_dev': 'Aantal standaarddeviaties voor de upper en lower bands',
            'sl_fixed_percent': 'Stop-loss als vast percentage',
            'tp_fixed_percent': 'Take-profit als vast percentage',
            'use_trailing_stop': 'Trailing stop activeren (true/false)',
            'trailing_stop_percent': 'Trailing stop als vast percentage',
            'trailing_activation_percent': 'Percentage prijsstijging voordat trailing stop wordt geactiveerd',
            'risk_per_trade': 'Risico per trade als percentage van portfolio (0.01 = 1%)',
            'confidence_level': 'Betrouwbaarheidsniveau voor VaR-berekening'}

        return descriptions

    @classmethod
    def get_performance_metrics(cls) -> List[str]:
        """Definieer performance metrics."""
        return ["sharpe_ratio", "calmar_ratio", "sortino_ratio", "win_rate",
                "total_return", "max_drawdown"]