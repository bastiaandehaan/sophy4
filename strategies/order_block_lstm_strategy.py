# strategies/order_block_lstm_strategy.py
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Controleer of TensorFlow geïnstalleerd is
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow")
    sys.exit(1)

from strategies.base_strategy import BaseStrategy
from strategies import register_strategy
from config import logger


class OrderBlock:
    """Representeert een order block zone in de markt."""

    def __init__(self, direction: int, time, high: float, low: float):
        self.direction = direction  # 1 = bullish, -1 = bearish
        self.time = time
        self.high = high
        self.low = low
        self.traded = False


@register_strategy
class OrderBlockLSTMStrategy(BaseStrategy):
    """
    Trading strategie gebaseerd op order blocks, Fibonacci retracements en LSTM-voorspellingen.

    Versie 1.1: Verbeterd door gebruik van alle drie Fibonacci niveaus (38.2%, 50%, 61.8%)
    """

    def __init__(self, window: int = 60, lstm_threshold: float = 0.0,
                 sl_fixed_percent: float = 0.01, tp_fixed_percent: float = 0.02,
                 use_trailing_stop: bool = False, trailing_stop_percent: float = 0.015,
                 risk_per_trade: float = 0.01, confidence_level: float = 0.95,
                 model_path: Optional[str] = None):
        """
        Initialiseer de OrderBlockLSTMStrategy.
        """
        super().__init__()
        self.window = window
        self.lstm_threshold = lstm_threshold
        self.sl_fixed_percent = sl_fixed_percent
        self.tp_fixed_percent = tp_fixed_percent
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_percent = trailing_stop_percent
        self.risk_per_trade = risk_per_trade
        self.confidence_level = confidence_level

        # Print versie-informatie direct naar console
        print("\n==== OrderBlockLSTM Strategie v1.1 ====")
        print(
            "Verbeterd met ondersteuning voor alle Fibonacci niveaus (38.2%, 50%, 61.8%)")
        print(
            f"Parameters: window={window}, lstm_threshold={lstm_threshold}, sl={sl_fixed_percent}, tp={tp_fixed_percent}")

        # Debug modus activeren (altijd aan in deze versie)
        self.debug_mode = True

        # Laad model, indien beschikbaar
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                logger.info(f"LSTM model geladen van {model_path}")
                print(f"LSTM model geladen van {model_path}")
            except Exception as e:
                logger.error(f"Kan LSTM model niet laden: {str(e)}")
                print(f"Waarschuwing: Kan LSTM model niet laden: {str(e)}")
        else:
            logger.warning(
                "Geen LSTM model opgegeven of gevonden, gebruik fallback logica")
            print(
                "Geen LSTM model opgegeven of gevonden, gebruik fallback logica (LSTM waarde = 0)")

    def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detecteer order blocks in de data met verbeterde logging.
        """
        obs = []
        logger.info(f"Data lengte voor order block detectie: {len(df)}")
        print(f"Data analyse: {len(df)} candles van {df.index[0]} tot {df.index[-1]}")

        # Debug info: Toon voorbeelden van enkele candles
        if self.debug_mode and len(df) >= 10:
            print("\nVoorbeeld van de laatste 5 candles:")
            for i in range(1, 6):
                if i <= len(df):
                    candle = df.iloc[-i]
                    print(
                        f"  Candle {df.index[-i]}: Open={candle['open']:.5f}, Close={candle['close']:.5f}, "
                        f"Bullish: {candle['close'] > candle['open']}")

        # Tellers voor debugging
        bullish_ob_count = 0
        bearish_ob_count = 0
        checked_candles = 0

        for i in range(len(df) - 3):
            checked_candles += 1

            # Zorg voor toegang tot 3 opeenvolgende candles
            if i + 2 >= len(df):
                continue

            c1, c2, c3 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2]

            # Detecteer Bullish OB: Bearish->Bearish->Bullish engulf
            if (c1['open'] > c1['close'] and  # C1 bearish
                    c2['open'] > c2['close'] and  # C2 bearish
                    c3['open'] < c3['close'] and  # C3 bullish
                    c3['open'] < c2['close']):  # C3 open onder C2 close (engulfing)

                obs.append(OrderBlock(1, c3.name, c3['high'], c3['low']))
                bullish_ob_count += 1

                # Gedetailleerde info over gevonden OB
                if bullish_ob_count % 10 == 1:  # Log af en toe een gevonden order block
                    print(
                        f"Bullish OB gevonden op {c3.name}: Range={c3['low']:.5f}-{c3['high']:.5f}")

            # Detecteer Bearish OB: Bullish->Bullish->Bearish engulf
            elif (c1['open'] < c1['close'] and  # C1 bullish
                  c2['open'] < c2['close'] and  # C2 bullish
                  c3['open'] > c3['close'] and  # C3 bearish
                  c3['open'] > c2['close']):  # C3 open boven C2 close (engulfing)

                obs.append(OrderBlock(-1, c3.name, c3['high'], c3['low']))
                bearish_ob_count += 1

                # Gedetailleerde info over gevonden OB
                if bearish_ob_count % 10 == 1:  # Log af en toe een gevonden order block
                    print(
                        f"Bearish OB gevonden op {c3.name}: Range={c3['low']:.5f}-{c3['high']:.5f}")

        print(f"\nOrder Block detectie resultaten:")
        print(f"  Bullish OBs gevonden: {bullish_ob_count}")
        print(f"  Bearish OBs gevonden: {bearish_ob_count}")
        print(f"  Totaal gevonden: {len(obs)}")

        logger.info(f"Order Block detectie resultaten:")
        logger.info(f"  Geanalyseerde candles: {checked_candles}")
        logger.info(f"  Bullish OBs gevonden: {bullish_ob_count}")
        logger.info(f"  Bearish OBs gevonden: {bearish_ob_count}")
        logger.info(f"  Totaal gevonden: {len(obs)}")

        # Bij geen order blocks, geef diagnose
        if len(obs) == 0 and len(df) > 5 and self.debug_mode:
            print("Geen order blocks gedetecteerd. Verificatie van candle patterns...")
            logger.warning(
                "Geen order blocks gedetecteerd. Verificatie van candle patterns:")

            # Tellers voor patronen die bijna een OB vormen
            almost_bullish = 0
            almost_bearish = 0
            missing_engulf_bullish = 0
            missing_engulf_bearish = 0

            # Voorbeelden van bijna-matchende patronen bewaren
            examples = []

            for i in range(len(df) - 3):
                if i + 2 >= len(df):
                    continue

                c1, c2, c3 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2]

                # Check patronen die bijna werken - voor diagnose
                if (c1['open'] > c1['close'] and c2['open'] > c2['close'] and c3[
                    'open'] < c3['close']):
                    almost_bullish += 1

                    # Check specifiek het engulfing criterium
                    if not (c3['open'] < c2['close']):
                        missing_engulf_bullish += 1

                        # Bewaar enkele voorbeelden
                        if len(examples) < 5:
                            examples.append(
                                {"index": i, "time": c3.name, "type": "bijna-bullish",
                                 "c2_close": c2['close'], "c3_open": c3['open'],
                                 "reden": "c3_open niet onder c2_close"})

                if (c1['open'] < c1['close'] and c2['open'] < c2['close'] and c3[
                    'open'] > c3['close']):
                    almost_bearish += 1

                    # Check specifiek het engulfing criterium
                    if not (c3['open'] > c2['close']):
                        missing_engulf_bearish += 1

                        # Bewaar enkele voorbeelden
                        if len(examples) < 10:
                            examples.append(
                                {"index": i, "time": c3.name, "type": "bijna-bearish",
                                 "c2_close": c2['close'], "c3_open": c3['open'],
                                 "reden": "c3_open niet boven c2_close"})

            print(
                f"Bijna-matchende patronen: {almost_bullish} bijna-bullish, {almost_bearish} bijna-bearish")
            print(
                f"  Ontbrekende engulfing: {missing_engulf_bullish} bullish, {missing_engulf_bearish} bearish")

            logger.info(
                f"Bijna-matchende patronen: {almost_bullish} bijna-bullish, {almost_bearish} bijna-bearish")
            logger.info(
                f"  Ontbrekende engulfing: {missing_engulf_bullish} bullish, {missing_engulf_bearish} bearish")

            # Toon enkele voorbeelden van bijna-matchende patronen
            if examples:
                print("Voorbeelden van bijna-matchende patronen:")
                for i, example in enumerate(examples[:5]):  # Toon max 5 voorbeelden
                    print(
                        f"  {i + 1}. {example['type']} op {example['time']}: {example['reden']} "
                        f"(c2_close={example['c2_close']:.5f}, c3_open={example['c3_open']:.5f})")

        return obs

    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float) -> Dict[
        str, float]:
        """
        Bereken Fibonacci retracement levels.
        """
        diff = swing_high - swing_low
        return {'61.8%': swing_high - 0.618 * diff, '50.0%': swing_high - 0.5 * diff,
                '38.2%': swing_high - 0.382 * diff}

    def prepare_lstm_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Bereid input voor voor het LSTM model.
        """
        if len(df) < self.window:
            logger.warning(
                f"Te weinig data voor LSTM (nodig: {self.window}, beschikbaar: {len(df)})")
            return np.zeros((1, self.window, 2))  # Fallback empty input

        # Controleer welke volume kolom beschikbaar is
        volume_column = None
        for possible_name in ['volume', 'tick_volume', 'real_volume']:
            if possible_name in df.columns:
                volume_column = possible_name
                break

        # Als geen volume kolom gevonden, maak een dummy kolom
        if volume_column is None:
            df['volume_dummy'] = 1.0
            volume_column = 'volume_dummy'

        # Normaliseer volume
        df[f'{volume_column}_norm'] = df[volume_column] / df[volume_column].rolling(
            window=20).mean().fillna(1)

        # Extract the most recent window of close and volume data
        recent_data = df.iloc[-self.window:]
        seq = [[row['close'], row.get(f'{volume_column}_norm', 1.0)] for _, row in
               recent_data.iterrows()]
        return np.array(seq).reshape(1, self.window, 2)

    def generate_signals(self, df: pd.DataFrame) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Genereer entries, sl_stop, tp_stop op basis van order blocks, Fibonacci en LSTM.

        Verbeterd in v1.1: Gebruikt nu alle drie Fibonacci niveaus (38.2%, 50%, 61.8%)
        """
        # Initialiseer lege Series met False/0
        entries = pd.Series(False, index=df.index)
        sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
        tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

        print(f"\nStart signaal generatie...")
        logger.info(
            f"Start signaal generatie voor {len(df)} candles, timeframe: {df.index[1] - df.index[0] if len(df) > 1 else 'onbekend'}")

        # Detecteer order blocks
        order_blocks = self.detect_order_blocks(df)
        if not order_blocks:
            print("Geen order blocks gedetecteerd, kan geen signalen genereren")
            logger.info("Geen order blocks gedetecteerd, kan geen signalen genereren")
            return entries, sl_stop, tp_stop

        # Bereid LSTM-voorspelling voor als model beschikbaar is
        lstm_pred = 0.0
        if self.model is not None:
            X_in = self.prepare_lstm_input(df)
            try:
                lstm_pred = self.model.predict(X_in, verbose=0)[0][0]
                print(
                    f"LSTM predictie: {lstm_pred:.4f}, threshold: {self.lstm_threshold}")
                logger.info(
                    f"LSTM predictie: {lstm_pred:.4f}, threshold: {self.lstm_threshold}")
            except Exception as e:
                print(f"LSTM predictie mislukt: {str(e)}")
                logger.error(f"LSTM predictie mislukt: {str(e)}")
        else:
            print("Geen LSTM model beschikbaar, gebruik standaard LSTM-waarde van 0")
            logger.warning(
                "Geen LSTM model beschikbaar, gebruik standaard LSTM-waarde van 0")

        # Loop door order blocks om signalen te genereren
        current_price = df['close'].iloc[-1]
        print(f"Huidige slotprijs: {current_price:.5f}")
        logger.info(f"Huidige prijs: {current_price:.5f}")

        # We kijken alleen naar Order Blocks in laatste 20% van de data
        time_threshold = df.index[int(len(df) * 0.8)]
        print(f"Tijdsfilter: Alleen OBs na {time_threshold} worden bekeken")
        logger.info(f"Tijdsfilter: Alleen OBs na {time_threshold} worden bekeken")

        # Variabelen voor diagnose
        signals_checked = 0
        in_fib_zone_count = 0
        lstm_ok_count = 0
        recent_ob_count = 0
        signals_generated = 0

        for ob in order_blocks:
            signals_checked += 1

            # Tijdsfilter toepassen - alleen recente OBs bekijken
            if ob.time < time_threshold:
                continue

            recent_ob_count += 1

            # Bereken Fibonacci niveaus
            past_data = df.loc[:ob.time]
            if past_data.empty:
                continue

            swing_high = past_data['high'].max()
            swing_low = past_data['low'].min()
            fib = self.calculate_fibonacci_levels(swing_high, swing_low)

            # Log elke 10e order block in detail
            if signals_checked % 10 == 1 or recent_ob_count <= 5:
                print(
                    f"\nOrder Block #{signals_checked}: {'Bullish' if ob.direction == 1 else 'Bearish'} op {ob.time}")
                print(f"  Prijsbereik: {ob.low:.5f} - {ob.high:.5f}")
                print(
                    f"  Fibonacci niveaus: 38.2%={fib['38.2%']:.5f}, 50.0%={fib['50.0%']:.5f}, 61.8%={fib['61.8%']:.5f}")

                logger.info(
                    f"Order Block #{signals_checked}: {'Bullish' if ob.direction == 1 else 'Bearish'} op {ob.time}")
                logger.info(f"  Prijsbereik: {ob.low:.5f} - {ob.high:.5f}")
                logger.info(
                    f"  Fibonacci niveaus: 61.8%={fib['61.8%']:.5f}, 50.0%={fib['50.0%']:.5f}, 38.2%={fib['38.2%']:.5f}")

            # VERBETERD: Check alle Fibonacci niveaus voor bullish signaal
            if ob.direction == 1:
                # Check of prijs in ENIG Fibonacci niveau ligt (38.2%, 50% of 61.8%)
                in_618_zone = fib['61.8%'] <= current_price <= ob.high
                in_50_zone = fib['50.0%'] <= current_price <= ob.high
                in_382_zone = fib['38.2%'] <= current_price <= ob.high

                # Prijs ligt in ten minste één van de Fibonacci zones
                in_fib_zone = in_618_zone or in_50_zone or in_382_zone

                # LSTM OK check blijft onveranderd
                lstm_ok = lstm_pred > self.lstm_threshold

                # Tellingen bijwerken voor diagnostiek
                if in_fib_zone:
                    in_fib_zone_count += 1
                if lstm_ok:
                    lstm_ok_count += 1

                if signals_checked % 10 == 1 or recent_ob_count <= 5:
                    print(
                        f"  Bullish signaal check: Prijs in Fib zone (38.2/50/61.8): {in_382_zone}/{in_50_zone}/{in_618_zone}, LSTM OK: {lstm_ok}")
                    logger.info(
                        f"  Bullish signaal check: Prijs in Fib zone: {in_fib_zone}, LSTM OK: {lstm_ok}")
                    logger.info(
                        f"  Details: current_price={current_price:.5f}, fib_zones={in_382_zone}/{in_50_zone}/{in_618_zone}, ob_high={ob.high:.5f}")

                if in_fib_zone and lstm_ok:
                    # Bullish signaal
                    entries.iloc[-1] = True
                    signals_generated += 1
                    print(f"✅ Bullish signaal gegenereerd op {df.index[-1]}")
                    logger.info(f"❗ Bullish signaal gegenereerd op {df.index[-1]}")

            # VERBETERD: Check alle Fibonacci niveaus voor bearish signaal
            elif ob.direction == -1:
                # Check of prijs in ENIG Fibonacci niveau ligt (38.2%, 50% of 61.8%)
                in_618_zone = ob.low <= current_price <= fib['61.8%']
                in_50_zone = ob.low <= current_price <= fib['50.0%']
                in_382_zone = ob.low <= current_price <= fib['38.2%']

                # Prijs ligt in ten minste één van de Fibonacci zones
                in_fib_zone = in_618_zone or in_50_zone or in_382_zone

                # LSTM OK check blijft onveranderd
                lstm_ok = lstm_pred < -self.lstm_threshold

                # Tellingen bijwerken voor diagnostiek
                if in_fib_zone:
                    in_fib_zone_count += 1
                if lstm_ok:
                    lstm_ok_count += 1

                if signals_checked % 10 == 1 or recent_ob_count <= 5:
                    print(
                        f"  Bearish signaal check: Prijs in Fib zone (38.2/50/61.8): {in_382_zone}/{in_50_zone}/{in_618_zone}, LSTM OK: {lstm_ok}")
                    logger.info(
                        f"  Bearish signaal check: Prijs in Fib zone: {in_fib_zone}, LSTM OK: {lstm_ok}")
                    logger.info(
                        f"  Details: current_price={current_price:.5f}, fib_zones={in_382_zone}/{in_50_zone}/{in_618_zone}, ob_low={ob.low:.5f}")

                # We genereren geen bearish signalen in deze implementatie (behouden)
                if in_fib_zone and lstm_ok:
                    logger.info(
                        f"  Bearish signaal gedetecteerd maar alleen long posities worden ondersteund")
                    print(
                        f"  Bearish signaal gedetecteerd maar alleen long posities worden ondersteund")

        # Log eindresultaat en diagnostiek
        signal_count = entries.sum()
        print(f"\nSignaal generatie resultaten:")
        print(f"  Order Blocks geanalyseerd: {signals_checked}")
        print(f"  Order Blocks na tijdsfilter: {recent_ob_count}")
        print(f"  Prijzen in Fibonacci zones: {in_fib_zone_count}")
        print(f"  LSTM voorspellingen boven threshold: {lstm_ok_count}")
        print(f"  Totaal aantal gegenereerde signalen: {signals_generated}")

        logger.info(f"Signaal generatie diagnostiek:")
        logger.info(f"  Totaal Order Blocks: {len(order_blocks)}")
        logger.info(f"  Recente Order Blocks (na tijdsfilter): {recent_ob_count}")
        logger.info(f"  Prijzen in Fibonacci zone: {in_fib_zone_count}")
        logger.info(f"  LSTM voorspellingen boven threshold: {lstm_ok_count}")
        logger.info(f"  Totaal aantal gegenereerde signalen: {signal_count}")

        return entries, sl_stop, tp_stop

    @classmethod
    def get_default_params(cls, timeframe: str = "H1") -> Dict[str, List[Any]]:
        """
        Geef default parameters voor de strategie.
        """
        return {'window': [30, 50, 60], 'lstm_threshold': [0.0, 0.1, 0.2],
                'sl_fixed_percent': [0.01, 0.015, 0.02],
                'tp_fixed_percent': [0.02, 0.03, 0.04],
                'use_trailing_stop': [True, False],
                'trailing_stop_percent': [0.01, 0.015, 0.02],
                'risk_per_trade': [0.005, 0.01, 0.02],
                'confidence_level': [0.90, 0.95, 0.99]}

    @classmethod
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """
        Beschrijf parameters voor documentatie.
        """
        return {'window': 'Aantal perioden voor LSTM input sequentie',
                'lstm_threshold': 'Drempelwaarde voor LSTM signalen (0 tot 1)',
                'sl_fixed_percent': 'Stop-loss als vast percentage',
                'tp_fixed_percent': 'Take-profit als vast percentage',
                'use_trailing_stop': 'Trailing stop activeren (true/false)',
                'trailing_stop_percent': 'Trailing stop als vast percentage',
                'risk_per_trade': 'Risico per trade als percentage van portfolio (0.01 = 1%)',
                'confidence_level': 'Betrouwbaarheidsniveau voor VaR-berekening',
                'model_path': 'Pad naar voorgetraind LSTM-model (.h5 bestand)'}