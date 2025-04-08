# strategies/bollong.py
import pandas as pd
import numpy as np
from typing import Tuple

from config import logger
from . import register_strategy  # Relatieve import om circulariteit te vermijden
from .base_strategy import BaseStrategy

def calculate_atr(df: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series]:
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=window).mean()
    return atr, tr

def calculate_adx(df: pd.DataFrame, tr: pd.Series = None, period: int = 14) -> pd.Series:
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    if tr is None:
        tr1 = abs(df['high'] - df['low'])
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = pd.Series(0.0, index=df.index)
    nonzero_mask = (plus_di + minus_di) > 0
    dx[nonzero_mask] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx

@register_strategy
class BollongStrategy(BaseStrategy):
    def __init__(self,
                 window: int = 50, std_dev: float = 2.0,
                 sl_method: str = "atr_based", sl_atr_mult: float = 2.0, sl_fixed_percent: float = 0.02,
                 tp_method: str = "atr_based", tp_atr_mult: float = 3.0, tp_fixed_percent: float = 0.03,
                 use_trailing_stop: bool = False, trailing_stop_method: str = "atr_based",
                 trailing_stop_atr_mult: float = 1.5, trailing_stop_percent: float = 0.015,
                 trailing_activation_percent: float = 0.01,
                 use_volume_filter: bool = False, volume_filter_periods: int = 20, volume_filter_mult: float = 1.5,
                 risk_per_trade: float = 0.01, max_positions: int = 3,
                 use_time_filter: bool = False, trading_hours: Tuple[int, int] = (9, 17),
                 min_adx: float = 20, use_adx_filter: bool = False):
        super().__init__()
        self.window = window
        self.std_dev = std_dev
        self.sl_method = sl_method
        self.sl_atr_mult = sl_atr_mult
        self.sl_fixed_percent = sl_fixed_percent
        self.tp_method = tp_method
        self.tp_atr_mult = tp_atr_mult
        self.tp_fixed_percent = tp_fixed_percent
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_method = trailing_stop_method
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.trailing_stop_percent = trailing_stop_percent
        self.trailing_activation_percent = trailing_activation_percent
        self.use_volume_filter = use_volume_filter
        self.volume_filter_periods = volume_filter_periods
        self.volume_filter_mult = volume_filter_mult
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions  # Blijft als parameter, maar niet gebruikt in backtest
        self.use_time_filter = use_time_filter
        self.trading_hours = trading_hours
        self.min_adx = min_adx
        self.use_adx_filter = use_adx_filter

    def validate_parameters(self) -> bool:
        if self.window < 5:
            raise ValueError("Window moet ten minste 5 zijn")
        if self.std_dev <= 0:
            raise ValueError("std_dev moet positief zijn")
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            raise ValueError("risk_per_trade moet tussen 0 en 0.1 (10%) liggen")
        return True

    def _calculate_stop(self, df: pd.DataFrame, atr: pd.Series, method: str, mult_key: str, fixed_key: str) -> pd.Series:
        return self.__dict__[mult_key] * atr / df['close'] if method == "atr_based" else pd.Series(self.__dict__[fixed_key], index=df.index)

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        has_datetime_index = hasattr(df.index, 'hour')

        sma = df['close'].rolling(window=self.window).mean()
        std = df['close'].rolling(window=self.window).std()
        upper_band = sma + (self.std_dev * std)

        atr, tr = calculate_atr(df)
        entries = df['close'] > upper_band  # Long-only

        if self.use_volume_filter and 'volume' in df.columns:
            avg_volume = df['volume'].rolling(window=self.volume_filter_periods).mean()
            volume_filter = df['volume'] > avg_volume * self.volume_filter_mult
            entries = entries & volume_filter

        if self.use_adx_filter:
            adx = calculate_adx(df, tr=tr)
            adx_filter = adx > self.min_adx
            entries = entries & adx_filter

        if self.use_time_filter:
            if not has_datetime_index:
                raise ValueError("Time filter vereist een datetime-index")
            time_filter = df.index.hour.isin(range(self.trading_hours[0], self.trading_hours[1] + 1))
            entries = entries & time_filter

        sl_stop = self._calculate_stop(df, atr, self.sl_method, 'sl_atr_mult', 'sl_fixed_percent')
        tp_stop = self._calculate_stop(df, atr, self.tp_method, 'tp_atr_mult', 'tp_fixed_percent')

        if self.use_trailing_stop:
            self.sl_trail = True
            trail_stop = self._calculate_stop(df, atr, self.trailing_stop_method, 'trailing_stop_atr_mult', 'trailing_stop_percent')
            sl_stop = trail_stop.clip(0.001, 0.999)
            if self.trailing_activation_percent > 0:
                logger.info(f"Trailing stop activeert na {self.trailing_activation_percent:.2%} prijsstijging")

        sl_stop = sl_stop.fillna(0.01)
        tp_stop = tp_stop.fillna(0.02)
        logger.info(f"Aantal LONG signalen: {entries.sum()}")
        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            'window': [20, 30, 40, 50, 60, 70, 80, 90, 100], 'std_dev': [1.5, 2.0, 2.5],
            'sl_method': ["fixed_percent"], 'sl_fixed_percent': [0.01, 0.015, 0.02, 0.025, 0.03],
            'tp_method': ["fixed_percent"], 'tp_fixed_percent': [0.02, 0.03, 0.04, 0.05, 0.06],
            'use_trailing_stop': [False, True], 'trailing_stop_percent': [0.01, 0.015, 0.02],
            'risk_per_trade': [0.01]
        }

    @classmethod
    def get_parameter_descriptions(cls) -> dict:
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
            'risk_per_trade': 'Risico per trade als percentage van portfolio (0.01 = 1%)'
        }

    @classmethod
    def get_performance_metrics(cls) -> list:
        return ["sharpe_ratio", "calmar_ratio", "sortino_ratio", "win_rate", "total_return", "max_drawdown"]