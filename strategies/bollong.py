# strategies/bollong.py
import pandas as pd

from config import logger
from strategies import register_strategy
from strategies.base_strategy import BaseStrategy


def calculate_atr(df, window=14):
    """
    Bereken de Average True Range (ATR) voor dynamische stop-loss en take-profit.
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr = pd.DataFrame(
        {'tr1': high - low, 'tr2': abs(high - close), 'tr3': abs(low - close)}).max(
        axis=1)
    return tr.rolling(window=window).mean()


@register_strategy
class BollongStrategy(BaseStrategy):
    """
    Bollinger Bands breakout strategie (long-only).
    Genereert signalen wanneer de prijs boven de upper band uitbreekt.
    """

    def __init__(self, # Signaal parameters
                 window=50, std_dev=2.0,

                 # Stop loss parameters
                 sl_method="atr_based",  # "atr_based" of "fixed_percent"
                 sl_atr_mult=2.0, sl_fixed_percent=0.02,

                 # Take profit parameters
                 tp_method="atr_based",  # "atr_based" of "fixed_percent"
                 tp_atr_mult=3.0, tp_fixed_percent=0.03,

                 # Trailing stop parameters
                 use_trailing_stop=False, trailing_stop_method="atr_based",
                 # "atr_based" of "fixed_percent"
                 trailing_stop_atr_mult=1.5, trailing_stop_percent=0.015,
                 trailing_activation_percent=0.01,  # Activeren na x% winst

                 # Volume filters
                 use_volume_filter=False, volume_filter_periods=20,
                 volume_filter_mult=1.5,

                 # Risk management
                 risk_per_trade=0.01,  # 1% risico per trade
                 max_positions=3,  # Max 3 posities tegelijk

                 # Time filters
                 use_time_filter=False, trading_hours=(9, 17),  # 9:00 - 17:00

                 # Andere filters
                 min_adx=20,  # Minimale ADX waarde voor trend sterkte
                 use_adx_filter=False):
        """
        Initialiseer de strategie met de gegeven parameters.
        """
        super().__init__()

        # Signaal parameters
        self.window = window
        self.std_dev = std_dev

        # Stop loss parameters
        self.sl_method = sl_method
        self.sl_atr_mult = sl_atr_mult
        self.sl_fixed_percent = sl_fixed_percent

        # Take profit parameters
        self.tp_method = tp_method
        self.tp_atr_mult = tp_atr_mult
        self.tp_fixed_percent = tp_fixed_percent

        # Trailing stop parameters
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_method = trailing_stop_method
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.trailing_stop_percent = trailing_stop_percent
        self.trailing_activation_percent = trailing_activation_percent

        # Volume filters
        self.use_volume_filter = use_volume_filter
        self.volume_filter_periods = volume_filter_periods
        self.volume_filter_mult = volume_filter_mult

        # Risk management
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions

        # Time filters
        self.use_time_filter = use_time_filter
        self.trading_hours = trading_hours

        # ADX filter
        self.min_adx = min_adx
        self.use_adx_filter = use_adx_filter

    def validate_parameters(self):
        """Controleer of parameters geldig zijn."""
        if self.window < 5:
            raise ValueError("Window moet ten minste 5 zijn")
        if self.std_dev <= 0:
            raise ValueError("std_dev moet positief zijn")
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            raise ValueError("risk_per_trade moet tussen 0 en 0.1 (10%) liggen")
        return True

    def calculate_adx(self, df, period=14):
        """Bereken Average Directional Index (ADX) voor trendsterkte."""
        # Implementeer ADX berekening als die nodig is
        # Dit is een placeholder
        return pd.Series(index=df.index, data=25.0)  # Placeholder waarde

    def generate_signals(self, df):
        """
        Genereer trading signalen op basis van Bollinger Band breakouts.

        Returns:
            tuple: (entries, sl_stop, tp_stop) - Series met entry signalen en stop percentages
        """
        # Bereken Bollinger Bands
        sma = df['close'].rolling(window=self.window).mean()
        std = df['close'].rolling(window=self.window).std()
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)

        # Bereken ATR
        atr = calculate_atr(df)

        # Basisconditie voor long-only entry signalen
        entries = df['close'] > upper_band

        # Volume filter toepassen indien ingeschakeld
        if self.use_volume_filter and 'volume' in df.columns:
            avg_volume = df['volume'].rolling(window=self.volume_filter_periods).mean()
            volume_filter = df['volume'] > avg_volume * self.volume_filter_mult
            entries = entries & volume_filter

        # ADX filter toepassen indien ingeschakeld
        if self.use_adx_filter:
            adx = self.calculate_adx(df)
            adx_filter = adx > self.min_adx
            entries = entries & adx_filter

        # Time filter toepassen indien ingeschakeld
        if self.use_time_filter:
            time_filter = df.index.hour.isin(
                range(self.trading_hours[0], self.trading_hours[1] + 1))
            entries = entries & time_filter

        # Stop-loss berekenen op basis van gekozen methode
        if self.sl_method == "atr_based":
            sl_stop = self.sl_atr_mult * atr / df['close']
        else:  # fixed_percent
            sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)

        # Take-profit berekenen op basis van gekozen methode
        if self.tp_method == "atr_based":
            tp_stop = self.tp_atr_mult * atr / df['close']
        else:  # fixed_percent
            tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

        # Trailing stop zou extra parameters naar VectorBT vereisen
        # Voor nu gebruiken we standaard sl/tp

        # Log aantal signalen
        logger.info(f"Aantal LONG signalen: {entries.sum()}")

        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls):
        """
        Default parameters voor grid search optimalisatie.
        """
        return {# Signaal parameters
            'window': range(20, 101, 10),  # 20, 30, 40, ..., 100
            'std_dev': [1.5, 2.0, 2.5, 3.0],  # Standaarddeviatie waarden

            # Stop loss parameters
            'sl_method': ["atr_based", "fixed_percent"],
            'sl_atr_mult': [1.0, 1.5, 2.0, 2.5, 3.0],
            'sl_fixed_percent': [0.01, 0.015, 0.02, 0.025, 0.03],

            # Take profit parameters
            'tp_method': ["atr_based", "fixed_percent"],
            'tp_atr_mult': [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            'tp_fixed_percent': [0.02, 0.03, 0.04, 0.05, 0.06],

            # Risk management
            'risk_per_trade': [0.005, 0.01, 0.015, 0.02]}

    @classmethod
    def get_parameter_descriptions(cls):
        """
        Beschrijft wat elke parameter doet voor documentatie.
        """
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
            'risk_per_trade': 'Risico per trade als percentage van portfolio (0.01 = 1%)'}

    @classmethod
    def get_performance_metrics(cls):
        """Definieer belangrijke metrics voor deze strategie."""
        return ["sharpe_ratio",  # Rendement/risico verhouding
            "calmar_ratio",  # Rendement/max drawdown
            "sortino_ratio",  # Downside risico maatstaf
            "win_rate",  # Percentage winstgevende trades
            "total_return",  # Totale winst
            "max_drawdown"  # Maximale drawdown (lager = beter)
        ]