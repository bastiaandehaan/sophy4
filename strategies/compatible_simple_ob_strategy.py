# strategies/simple_ob_strategy.py
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy
from strategies import register_strategy


class OrderBlock:
    """Representeert een order block zone in de markt."""

    def __init__(self, direction: int, time, high: float, low: float):
        self.direction = direction  # 1 = bullish, -1 = bearish
        self.time = time
        self.high = high
        self.low = low
        self.traded = False


@register_strategy
class SimpleOBStrategy(BaseStrategy):
    """
    Vereenvoudigde Order Block strategie zonder LSTM en met soepelere criteria.
    Directe console output voor debugging.
    """

    def __init__(self, window: int = 20, sl_fixed_percent: float = 0.01,
                 tp_fixed_percent: float = 0.02, risk_per_trade: float = 0.01,
                 confidence_level: float = 0.95, use_trailing_stop: bool = False,
                 trailing_stop_percent: float = 0.015,
                 trailing_stop_activation_percent: float = 0.01,
                 model_path: Optional[str] = None,
                 **kwargs):  # Accept any other parameters that might be passed
        """
        Initialiseer de SimpleOBStrategy - compatibel met alle bestaande parameters.
        """
        super().__init__()
        self.window = window
        self.sl_fixed_percent = sl_fixed_percent
        self.tp_fixed_percent = tp_fixed_percent
        self.risk_per_trade = risk_per_trade
        self.confidence_level = confidence_level
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_percent = trailing_stop_percent

        # We negeren model_path, we maken geen gebruik van LSTM

        # Standaard NIET tijdsfilter gebruiken
        self.use_time_filter = False

        print("\n===== Simple Order Block Strategy Geactiveerd =====")
        print(
            f"Parameters: window={window}, sl={sl_fixed_percent}, tp={tp_fixed_percent}")
        print(f"Tijdsfilter: {'AAN' if self.use_time_filter else 'UIT'}")
        print(f"Trailing stop: {'AAN' if use_trailing_stop else 'UIT'}")

    def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detecteer order blocks met SOEPELERE criteria:
        - Bullish OB: Bearish candle gevolgd door Bullish candle
        - Bearish OB: Bullish candle gevolgd door Bearish candle
        """
        obs = []
        print(f"Data analyse: {len(df)} candles van {df.index[0]} tot {df.index[-1]}")

        bullish_count = 0
        bearish_count = 0

        # Tonen van enkele voorbeeld-candles
        print("\nVoorbeeld candles (laatste 5):")
        for i in range(1, 6):
            if i <= len(df):
                candle = df.iloc[-i]
                print(
                    f"  Candle {df.index[-i]}: O={candle['open']:.5f}, C={candle['close']:.5f}, Bullish: {candle['close'] > candle['open']}")

        # Soepelere criteria voor order blocks
        for i in range(len(df) - 2):
            if i + 1 >= len(df):  # Veiligheidscheck
                continue

            c1, c2 = df.iloc[i], df.iloc[i + 1]

            # BULLISH ORDER BLOCK: Bearish gevolgd door Bullish met significante beweging
            if (c1['open'] > c1['close'] and  # C1 bearish
                    c2['open'] < c2['close'] and  # C2 bullish
                    abs(c2['close'] - c2['open']) / c2[
                        'open'] > 0.0005):  # Significante beweging (0.05%)

                obs.append(OrderBlock(1, c2.name, c2['high'], c2['low']))
                bullish_count += 1

                # Log af en toe een gevonden order block
                if bullish_count % 10 == 1:
                    print(
                        f"Bullish OB gevonden op {c2.name}: Range={c2['low']:.5f}-{c2['high']:.5f}")

            # BEARISH ORDER BLOCK: Bullish gevolgd door Bearish met significante beweging
            elif (c1['open'] < c1['close'] and  # C1 bullish
                  c2['open'] > c2['close'] and  # C2 bearish
                  abs(c2['open'] - c2['close']) / c2[
                      'open'] > 0.0005):  # Significante beweging (0.05%)

                obs.append(OrderBlock(-1, c2.name, c2['high'], c2['low']))
                bearish_count += 1

                # Log af en toe een gevonden order block
                if bearish_count % 10 == 1:
                    print(
                        f"Bearish OB gevonden op {c2.name}: Range={c2['low']:.5f}-{c2['high']:.5f}")

        # Resultaten rapporteren
        print(f"\nOrder Block detectie resultaten:")
        print(f"  Bullish OBs gevonden: {bullish_count}")
        print(f"  Bearish OBs gevonden: {bearish_count}")
        print(f"  Totaal gevonden: {len(obs)}")

        return obs

    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float) -> Dict[
        str, float]:
        """
        Bereken Fibonacci retracement levels.
        """
        diff = swing_high - swing_low
        return {'61.8%': swing_high - 0.618 * diff, '50.0%': swing_high - 0.5 * diff,
            '38.2%': swing_high - 0.382 * diff}

    def generate_signals(self, df: pd.DataFrame) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Genereer handelsignalen op basis van Order Blocks en Fibonacci zones.
        ZONDER LSTM en met optioneel tijdsfilter.
        """
        # Initialiseer lege Series
        entries = pd.Series(False, index=df.index)
        sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
        tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

        print(f"\nStart signaal generatie...")

        # Detecteer order blocks met soepelere criteria
        order_blocks = self.detect_order_blocks(df)
        if not order_blocks:
            print("Geen order blocks gedetecteerd, forceer signaal op laatste candle")
            entries.iloc[-1] = True
            print(f"✅ Geforceerd signaal op {df.index[-1]}")
            return entries, sl_stop, tp_stop

        # Huidige prijs voor signaal evaluatie
        current_price = df['close'].iloc[-1]
        print(f"Huidige slotprijs: {current_price:.5f}")

        # Tijdsfilter alleen indien ingeschakeld
        time_threshold = df.index[0]  # Standaard: geen filtering
        if self.use_time_filter:
            time_threshold = df.index[int(len(df) * 0.8)]  # Laatste 20%
            print(f"Tijdsfilter actief: Alleen OBs na {time_threshold} worden gebruikt")

        # Variabelen voor diagnostiek
        signals_checked = 0
        in_fib_zone_count = 0
        recent_ob_count = 0
        signals_generated = 0

        # Loop door alle order blocks
        for ob in order_blocks:
            signals_checked += 1

            # Tijdsfilter toepassen indien ingeschakeld
            if self.use_time_filter and ob.time < time_threshold:
                continue

            recent_ob_count += 1

            # Bereken Fibonacci niveaus met high/low tot aan order block
            past_data = df.loc[:ob.time]
            if past_data.empty:
                continue

            swing_high = past_data['high'].max()
            swing_low = past_data['low'].min()
            fib = self.calculate_fibonacci_levels(swing_high, swing_low)

            # Elke 10e order block in detail loggen
            if signals_checked % 10 == 1:
                print(
                    f"\nOrder Block #{signals_checked}: {'Bullish' if ob.direction == 1 else 'Bearish'} op {ob.time}")
                print(f"  Prijsbereik: {ob.low:.5f} - {ob.high:.5f}")
                print(
                    f"  Fibonacci niveaus: 38.2%={fib['38.2%']:.5f}, 50.0%={fib['50.0%']:.5f}, 61.8%={fib['61.8%']:.5f}")

            # SOEPELER: Check alle Fibonacci niveaus (38.2%, 50%, 61.8%)
            if ob.direction == 1:  # Bullish OB
                # Check of prijs tussen ENIG Fibonacci niveau en ob.high ligt
                in_38_zone = fib['38.2%'] <= current_price <= ob.high
                in_50_zone = fib['50.0%'] <= current_price <= ob.high
                in_618_zone = fib['61.8%'] <= current_price <= ob.high
                in_fib_zone = in_38_zone or in_50_zone or in_618_zone

                if signals_checked % 10 == 1:
                    print(
                        f"  Bullish check: Prijs in Fib zone (38.2/50/61.8): {in_38_zone}/{in_50_zone}/{in_618_zone}")

                if in_fib_zone:
                    in_fib_zone_count += 1
                    entries.iloc[-1] = True
                    signals_generated += 1
                    print(f"✅ Bullish signaal gegenereerd op {df.index[-1]}")

            elif ob.direction == -1:  # Bearish OB
                # Check of prijs tussen ob.low en ENIG Fibonacci niveau ligt
                in_38_zone = ob.low <= current_price <= fib['38.2%']
                in_50_zone = ob.low <= current_price <= fib['50.0%']
                in_618_zone = ob.low <= current_price <= fib['61.8%']
                in_fib_zone = in_38_zone or in_50_zone or in_618_zone

                if signals_checked % 10 == 1:
                    print(
                        f"  Bearish check: Prijs in Fib zone (38.2/50/61.8): {in_38_zone}/{in_50_zone}/{in_618_zone}")

                if in_fib_zone:
                    in_fib_zone_count += 1
                    entries.iloc[-1] = True
                    signals_generated += 1
                    print(f"✅ Bearish signaal gegenereerd op {df.index[-1]}")

        # Als geen signalen gegenereerd maar wel recente order blocks, forceer een signaal op laatste candle
        if signals_generated == 0 and recent_ob_count > 0:
            entries.iloc[-1] = True
            signals_generated = 1
            print(
                f"⚠️ Geen signalen op basis van criteria, maar forceer signaal op laatste candle ({df.index[-1]})")

        # Resultaten rapporteren
        print(f"\nSignaal generatie resultaten:")
        print(f"  Order Blocks geanalyseerd: {signals_checked}")
        print(f"  Order Blocks na tijdsfilter: {recent_ob_count}")
        print(f"  Prijzen in Fibonacci zones: {in_fib_zone_count}")
        print(f"  Totaal signalen gegenereerd: {signals_generated}")

        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """Geef default parameters voor de strategie."""
        return {'window': [20, 30, 40], 'sl_fixed_percent': [0.01, 0.015, 0.02],
                'tp_fixed_percent': [0.02, 0.03, 0.04],
                'risk_per_trade': [0.005, 0.01, 0.02],
                'confidence_level': [0.90, 0.95, 0.99],
                'use_trailing_stop': [False, True],
                'trailing_stop_percent': [0.01, 0.015, 0.02]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """Beschrijf parameters voor documentatie."""
        return {'window': 'Aantal perioden voor historische berekeningen',
                'sl_fixed_percent': 'Stop-loss als vast percentage',
                'tp_fixed_percent': 'Take-profit als vast percentage',
                'risk_per_trade': 'Risico per trade als percentage van portfolio (0.01 = 1%)',
                'confidence_level': 'Betrouwbaarheidsniveau voor VaR-berekening',
                'use_trailing_stop': 'Trailing stop activeren (true/false)',
                'trailing_stop_percent': 'Trailing stop als vast percentage'}