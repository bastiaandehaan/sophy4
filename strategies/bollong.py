# strategies/bollong.py
import json
from typing import Tuple, Optional, Dict, List, Any

import pandas as pd

from config import logger
from risk.risk_management import RiskManager
from utils.indicator_utils import calculate_bollinger_bands
from . import register_strategy
from .base_strategy import BaseStrategy


def calculate_atr(df: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Bereken ATR en True Range."""
    high: pd.Series = df['high']
    low: pd.Series = df['low']
    close: pd.Series = df['close'].shift(1)

    tr1: pd.Series = high - low
    tr2: pd.Series = (high - close).abs()
    tr3: pd.Series = (low - close).abs()
    tr: pd.Series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr: pd.Series = tr.rolling(window=window).mean()
    return atr, tr


def calculate_adx(df: pd.DataFrame, tr: Optional[pd.Series] = None,
                  period: int = 14) -> pd.Series:
    """Bereken ADX."""
    plus_dm: pd.Series = df['high'].diff()
    minus_dm: pd.Series = df['low'].diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    if tr is None:
        tr1: pd.Series = abs(df['high'] - df['low'])
        tr2: pd.Series = abs(df['high'] - df['close'].shift(1))
        tr3: pd.Series = abs(df['low'] - df['close'].shift(1))
        tr: pd.Series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr: pd.Series = tr.rolling(window=period).mean()
    plus_di: pd.Series = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di: pd.Series = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx: pd.Series = pd.Series(0.0, index=df.index)
    nonzero_mask: pd.Series = (plus_di + minus_di) > 0
    dx[nonzero_mask] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx: pd.Series = dx.rolling(window=period).mean()

    return adx


@register_strategy
class BollongStrategy(BaseStrategy):
    def __init__(self, window: int = 50, std_dev: float = 2.0,
                 sl_method: str = "atr_based", sl_atr_mult: float = 2.0,
                 sl_fixed_percent: float = 0.02, tp_method: str = "atr_based",
                 tp_atr_mult: float = 3.0, tp_fixed_percent: float = 0.03,
                 use_trailing_stop: bool = True,  # Changed to True
                 trailing_stop_method: str = "atr_based",
                 trailing_stop_atr_mult: float = 1.5,
                 trailing_stop_percent: float = 0.015,
                 trailing_activation_percent: float = 0.01,
                 use_volume_filter: bool = False, volume_filter_periods: int = 20,
                 volume_filter_mult: float = 1.5, risk_per_trade: float = 0.01,
                 max_positions: int = 3, use_time_filter: bool = False,
                 trading_hours: Tuple[int, int] = (9, 17), min_adx: float = 20,
                 use_adx_filter: bool = False, confidence_level: float = 0.95):
        super().__init__()
        self.window: int = window
        self.std_dev: float = std_dev
        self.sl_method: str = sl_method
        self.sl_atr_mult: float = sl_atr_mult
        self.sl_fixed_percent: float = sl_fixed_percent
        self.tp_method: str = tp_method
        self.tp_atr_mult: float = tp_atr_mult
        self.tp_fixed_percent: float = tp_fixed_percent
        self.use_trailing_stop: bool = use_trailing_stop
        self.trailing_stop_method: str = trailing_stop_method
        self.trailing_stop_atr_mult: float = trailing_stop_atr_mult
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
        if self.sl_atr_mult <= 0 or self.tp_atr_mult <= 0:
            raise ValueError("ATR multipliers moeten positief zijn")
        return True

    def _calculate_stop(self, df: pd.DataFrame, atr: pd.Series, method: str,
                        mult_key: str, fixed_key: str) -> pd.Series:
        """Bereken stop-waarde."""
        return self.__dict__[mult_key] * atr / df[
            'close'] if method == "atr_based" else pd.Series(self.__dict__[fixed_key],
                                                             index=df.index)

    def generate_signals(self, df: pd.DataFrame) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """Genereer trading signalen."""
        has_datetime_index: bool = hasattr(df.index, 'hour')

        risk_manager: RiskManager = RiskManager(confidence_level=self.confidence_level,
                                                max_risk=self.risk_per_trade)
        returns: pd.Series = df['close'].pct_change().dropna()
        from config import INITIAL_CAPITAL, PIP_VALUE
        size: float = risk_manager.calculate_position_size(INITIAL_CAPITAL, returns,
                                                           pip_value=PIP_VALUE)

        # Bereken Bollinger Bands met self.window en self.std_dev
        upper_band, sma, lower_band = calculate_bollinger_bands(df['close'],
                                                                window=self.window,
                                                                std_dev=self.std_dev)

        # Bereken de bandbreedte
        band_width = upper_band - lower_band
        price_position = (df['close'] - lower_band) / band_width
        entries = price_position > 0.7  # Prijs in de bovenste 30% van de bandbreedte (was 0.8)

        # Volatiliteitsfilter: geen nieuwe signalen als ATR te hoog is
        atr, tr = calculate_atr(df)
        avg_atr = atr.rolling(window=20).mean()
        volatility_filter = atr < avg_atr * 3.0  # Relaxed from 2.0 to 3.0
        entries = entries & volatility_filter

        # Bear market-filter: alleen signalen in een bull market
        long_sma = df['close'].rolling(window=100).mean()  # Increased from 50 to 100
        bull_market = df[
                          'close'] > long_sma  # Alleen signalen als prijs boven lange SMA
        entries = entries & bull_market

        logger.info(f"Gemiddeld aantal signalen: {entries.sum()} over {len(df)} bars")

        # Bestaande filters blijven intact
        if self.use_volume_filter and 'volume' in df.columns:
            avg_volume: pd.Series = df['volume'].rolling(
                window=self.volume_filter_periods).mean()
            volume_filter: pd.Series = df[
                                           'volume'] > avg_volume * self.volume_filter_mult
            entries = entries & volume_filter

        if self.use_adx_filter:
            atr, tr = calculate_atr(df)
            adx: pd.Series = calculate_adx(df, tr=tr)
            adx_filter: pd.Series = adx > self.min_adx
            entries = entries & adx_filter

        if self.use_time_filter:
            if not has_datetime_index:
                raise ValueError("Time filter vereist een datetime-index")
            time_filter: pd.Series = df.index.hour.isin(
                range(self.trading_hours[0], self.trading_hours[1] + 1))
            entries = entries & time_filter

        # Bereken stops met strakkere waarden
        atr, tr = calculate_atr(df)
        sl_stop: pd.Series = self._calculate_stop(df, atr, self.sl_method,
                                                  'sl_atr_mult', 'sl_fixed_percent')
        tp_stop: pd.Series = self._calculate_stop(df, atr, self.tp_method,
                                                  'tp_atr_mult', 'tp_fixed_percent')

        if self.use_trailing_stop:
            self.sl_trail = True
            trail_stop: pd.Series = self._calculate_stop(df, atr,
                                                         self.trailing_stop_method,
                                                         'trailing_stop_atr_mult',
                                                         'trailing_stop_percent')
            sl_stop = trail_stop.clip(0.001, 0.999)
            if self.trailing_activation_percent > 0:
                logger.info(
                    f"Trailing stop activeert na {self.trailing_activation_percent:.2%} prijsstijging")

        sl_stop = sl_stop.fillna(0.015)  # Strakkere stop-loss: 1,5%
        tp_stop = tp_stop.fillna(0.03)  # Kleinere take-profit: 3%
        logger.info(f"Aantal LONG signalen: {entries.sum()}")
        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """Geef default parameters terug gebaseerd op timeframe."""
        with open("timeframe_config.json", "r") as f:
            config = json.load(f)

        tf_config = config.get(timeframe,
                               config["H1"])  # Default to H1 if timeframe not found
        return {'window': tf_config["window_range"],
            'std_dev': tf_config["std_dev_range"], 'sl_method': ["fixed_percent"],
            'sl_fixed_percent': tf_config["sl_fixed_percent_range"],
            'tp_method': ["fixed_percent"],
            'tp_fixed_percent': tf_config["tp_fixed_percent_range"],
            'use_trailing_stop': [True],  # Only test with trailing stop enabled
            'trailing_stop_percent': [0.01, 0.015, 0.02],  # Wider range
            'risk_per_trade': [0.005, 0.01], 'confidence_level': [0.90, 0.95, 0.99]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """Beschrijf parameters."""
        return {
            'window': 'Aantal perioden voor Bollinger Bands (voortschrijdend gemiddelde)',
            'std_dev': 'Aantal standaarddeviaties voor de upper en lower bands',
            'sl_method': 'Stop-loss methode: "atr_based" of "fixed_percent"',
            'sl_atr_mult': 'Stop-loss als factor van ATR (als sl_method="atr_based")',
            'sl_fixed_percent': 'Stop-loss als vast percentage (als sl_method="fixed_percent")',
            'tp_method': 'Take-profit methode: "atr_based" of "fixed_percent"',
            'tp_atr_mult': 'Take-profit als factor van ATR (als tp_method="atr_based")',
            'tp_fixed_percent': 'Take-profit als vast percentage (als tp_method="fixed_percent")',
            'use_trailing_stop': 'Trailing stop activeren (true/false)',
            'trailing_stop_method': 'Methode voor trailing stop: "atr_based" of "fixed_percent"',
            'trailing_stop_percent': 'Trailing stop als vast percentage',
            'trailing_stop_atr_mult': 'Trailing stop als factor van ATR',
            'trailing_activation_percent': 'Percentage prijsstijging voordat trailing stop wordt geactiveerd',
            'risk_per_trade': 'Risico per trade als percentage van portfolio (0.01 = 1%)',
            'confidence_level': 'Betrouwbaarheidsniveau voor VaR-berekening'}

    @classmethod
    def get_performance_metrics(cls) -> List[str]:
        """Definieer performance metrics."""
        return ["sharpe_ratio", "calmar_ratio", "sortino_ratio", "win_rate",
                "total_return", "max_drawdown"]