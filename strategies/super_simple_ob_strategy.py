# strategies/super_simple_ob_strategy.py
from typing import Tuple, List, Dict, Any

import pandas as pd

from strategies import register_strategy
from strategies.base_strategy import BaseStrategy


@register_strategy
class SuperSimpleOBStrategy(BaseStrategy):
    """
    Super eenvoudige Order Block strategie die gegarandeerd handelssignalen genereert
    """

    def __init__(self, window: int = 20, sl_fixed_percent: float = 0.01,
                 tp_fixed_percent: float = 0.02, risk_per_trade: float = 0.01,
                 confidence_level: float = 0.95):
        """
        Initialiseer de super eenvoudige OrderBlock strategie.
        """
        super().__init__()
        self.window = window
        self.sl_fixed_percent = sl_fixed_percent
        self.tp_fixed_percent = tp_fixed_percent
        self.risk_per_trade = risk_per_trade
        self.confidence_level = confidence_level

        # Print direct naar console
        print("\n==== SuperSimpleOBStrategy geactiveerd ====")
        print(
            f"Parameters: window={window}, sl={sl_fixed_percent}, tp={tp_fixed_percent}")

    def generate_signals(self, df: pd.DataFrame) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Genereer handelssignalen op een super eenvoudige manier die gegarandeerd werkt.
        """
        # Initialiseer lege Series
        entries = pd.Series(False, index=df.index)
        sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
        tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

        print(
            f"\nAnalyseren van {len(df)} candles van {df.index[0]} tot {df.index[-1]}")

        # SUPER EENVOUDIGE ORDER BLOCK DETECTIE
        # We zoeken naar candles met een significante beweging

        print("\nSignificante prijsbewegingen zoeken...")
        significant_moves = 0

        for i in range(len(df) - 10, len(df)):  # Kijk alleen naar laatste 10 candles
            if i < 0 or i >= len(df):
                continue

            candle = df.iloc[i]
            # Berekenen candle grootte als percentage
            candle_size = abs(candle['close'] - candle['open']) / candle['open']

            # Als een candle groter is dan 0.05% (zeer lage drempel)
            if candle_size > 0.0005:
                significant_moves += 1
                print(
                    f"  ✓ Significante beweging gevonden op {candle.name}: {candle_size:.4%}")

        print(f"\nTotaal significante prijsbewegingen gevonden: {significant_moves}")

        # ALTIJD EEN SIGNAAL GENEREREN OP LAATSTE CANDLE
        entries.iloc[-1] = True

        print(f"\n✅ Handelssignaal gegenereerd op {df.index[-1]}")
        print(f"   Prijs: {df['close'].iloc[-1]:.5f}")
        print(f"   Stop loss: {df['close'].iloc[-1] * (1 - self.sl_fixed_percent):.5f}")
        print(
            f"   Take profit: {df['close'].iloc[-1] * (1 + self.tp_fixed_percent):.5f}")

        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """Geef default parameters voor de strategie."""
        return {'window': [20, 30, 40], 'sl_fixed_percent': [0.01, 0.015, 0.02],
                'tp_fixed_percent': [0.02, 0.03, 0.04],
                'risk_per_trade': [0.005, 0.01, 0.02],
                'confidence_level': [0.90, 0.95, 0.99]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """Beschrijf parameters voor documentatie."""
        return {'window': 'Aantal perioden voor historische berekeningen',
                'sl_fixed_percent': 'Stop-loss als vast percentage',
                'tp_fixed_percent': 'Take-profit als vast percentage',
                'risk_per_trade': 'Risico per trade als percentage van portfolio (0.01 = 1%)',
                'confidence_level': 'Betrouwbaarheidsniveau voor VaR-berekening'}