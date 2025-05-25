# strategies/bollong.py - FIXED VERSION
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

from strategies import register_strategy
from strategies.base_strategy import BaseStrategy
from utils.indicator_utils import calculate_bollinger_bands  # 🚀 Use unified version

# Logger configureren
logger = logging.getLogger(__name__)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Bereken Average True Range (ATR).

    Args:
        df: DataFrame met OHLC data
        window: Aantal periods voor ATR berekening

    Returns:
        Tuple van (ATR, TR series)
    """
    # Bereken True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.fillna(tr1, inplace=True)

    # Bereken ATR
    atr = tr.rolling(window=window).mean()

    # Vul NaN waarden in
    atr.fillna(tr, inplace=True)

    return atr, tr


def calculate_adx(df: pd.DataFrame, window: int = 14,
                  tr: Optional[pd.Series] = None) -> pd.Series:
    """
    Bereken Average Directional Index (ADX).

    Args:
        df: DataFrame met OHLC data
        window: Aantal periods voor ADX berekening
        tr: Optionele True Range series (voor performance)

    Returns:
        ADX series
    """
    # Bereken True Range als niet meegegeven
    if tr is None:
        _, tr = calculate_atr(df, window)

    # Bereken +DM en -DM
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()

    # Voorwaarden voor +DM en -DM
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
    minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)

    # Bereken +DI en -DI
    plus_di = 100 * plus_dm.rolling(window).mean() / tr.rolling(window).mean()
    minus_di = 100 * minus_dm.rolling(window).mean() / tr.rolling(window).mean()

    # Bereken DX en ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window).mean()

    # Vul NaN waarden in (eenvoudige backfill)
    adx.fillna(0, inplace=True)

    return adx


@register_strategy
class BollongStrategy(BaseStrategy):
    def __init__(self, symbol: str = "EURUSD", window: int = 50, std_dev: float = 2.0,
                 sl_method: str = "atr_based", sl_atr_mult: float = 2.0,
                 sl_fixed_percent: float = 0.02, tp_method: str = "atr_based",
                 tp_atr_mult: float = 3.0, tp_fixed_percent: float = 0.03,
                 use_trailing_stop: bool = True,
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

        # Core parameters
        self.symbol: str = symbol
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

        logger.info(f"BollongStrategy initialized: window={window}, std_dev={std_dev}")

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
        if method == "atr_based":
            return self.__dict__[mult_key] * atr / df['close']
        else:
            return pd.Series(self.__dict__[fixed_key], index=df.index)

    def generate_signals(self, df: pd.DataFrame,
                         current_capital: Optional[float] = None) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """Genereer trading signalen."""

        logger.info(f"Generating Bollinger signals for {len(df)} bars")

        has_datetime_index: bool = hasattr(df.index, 'hour')

        # 🚀 FIXED: Simple risk calculation without complex imports
        # No need for complex risk manager in signal generation

        # Bereken Bollinger Bands - NU UNIFIED
        upper_band, sma, lower_band = calculate_bollinger_bands(df['close'],
            window=self.window, std_dev=self.std_dev)

        # Bereken de bandbreedte en positie
        band_width = upper_band - lower_band
        price_position = (df['close'] - lower_band) / band_width

        # 🎯 Entry signalen: prijs in bovenste deel van Bollinger Bands
        entries = price_position > 0.7  # Prijs in de bovenste 30% van de bandbreedte

        # Volatiliteitsfilter: geen nieuwe signalen als ATR te hoog is
        atr, tr = calculate_atr(df)
        avg_atr = atr.rolling(window=20).mean()
        volatility_filter = atr < avg_atr * 3.0  # Relaxed volatility filter
        entries = entries & volatility_filter

        # Bull market filter: alleen signalen boven lange SMA
        long_sma = df['close'].rolling(window=100).mean()
        bull_market = df['close'] > long_sma
        entries = entries & bull_market

        # Volume filter (optioneel)
        if self.use_volume_filter and 'tick_volume' in df.columns:
            avg_volume = df['tick_volume'].rolling(
                window=self.volume_filter_periods).mean()
            volume_filter = df['tick_volume'] > avg_volume * self.volume_filter_mult
            entries = entries & volume_filter

        # ADX filter (optioneel)
        if self.use_adx_filter:
            adx = calculate_adx(df, tr=tr)
            adx_filter = adx > self.min_adx
            entries = entries & adx_filter

        # Time filter (optioneel)
        if self.use_time_filter:
            if not has_datetime_index:
                raise ValueError("Time filter vereist een datetime-index")
            time_filter = df.index.hour.isin(
                range(self.trading_hours[0], self.trading_hours[1] + 1))
            entries = entries & time_filter

        # Bereken stops
        atr, tr = calculate_atr(df)
        sl_stop = self._calculate_stop(df, atr, self.sl_method, 'sl_atr_mult',
                                       'sl_fixed_percent')
        tp_stop = self._calculate_stop(df, atr, self.tp_method, 'tp_atr_mult',
                                       'tp_fixed_percent')

        # Trailing stop handling
        if self.use_trailing_stop:
            trail_stop = self._calculate_stop(df, atr, self.trailing_stop_method,
                                              'trailing_stop_atr_mult',
                                              'trailing_stop_percent')
            sl_stop = trail_stop.clip(0.001, 0.999)
            if self.trailing_activation_percent > 0:
                logger.info(
                    f"Trailing stop activeert na {self.trailing_activation_percent:.2%}")

        # Fill NaN values with defaults
        sl_stop = sl_stop.fillna(self.sl_fixed_percent)
        tp_stop = tp_stop.fillna(self.tp_fixed_percent)

        # Convert to integer signals
        entries = entries.fillna(False).astype(int)

        # Log results
        num_signals = entries.sum()
        logger.info(f"Bollinger Strategy: {num_signals} signals generated")
        logger.info(f"Signal rate: {num_signals / len(df) * 100:.1f}% of bars")

        return entries, sl_stop, tp_stop

    @classmethod
    def get_performance_metrics(cls) -> List[str]:
        """Definieer performance metrics."""
        return ["sharpe_ratio", "calmar_ratio", "sortino_ratio", "win_rate",
                "total_return", "max_drawdown"]

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """Get default parameters for optimization."""
        return {'window': [20, 30, 50, 60], 'std_dev': [1.5, 2.0, 2.5, 3.0],
            'sl_fixed_percent': [0.015, 0.02, 0.025],
            'tp_fixed_percent': [0.03, 0.04, 0.05],
            'risk_per_trade': [0.01, 0.015, 0.02]}