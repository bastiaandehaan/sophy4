#!/usr/bin/env python
# run_live_trading.py
"""
Sophy4 Live Trading Tool

Dit script voert een live trading strategie uit op basis van geoptimaliseerde parameters.
Het ondersteunt flexibele configuratie via YAML, caching, en uitgebreide notificaties.

Gebruik:
    python run_live_trading.py --strategy BollongStrategy --symbol GER40.cash --timeframe D1 --params_file results/optimized.json
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd
import yaml

# Voeg projectroot toe aan pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from strategies import get_strategy
from utils.data_utils import fetch_historical_data
from live.live_trading import run_live_trading
from monitor.monitor import run_monitor_loop
from config import logger, OUTPUT_DIR

# Globale cache voor data
data_cache = {}

# Timeframe mapping
TIMEFRAME_INTERVALS = {
    'M1': timedelta(minutes=1), 'M5': timedelta(minutes=5), 'M15': timedelta(minutes=15),
    'M30': timedelta(minutes=30), 'H1': timedelta(hours=1), 'H4': timedelta(hours=4),
    'D1': timedelta(days=1), 'W1': timedelta(weeks=1),
}


def load_config(config_file="config.yaml") -> Dict:
    """Laad standaardconfiguratie uit een YAML-bestand."""
    default_config = {
        'risk_per_trade': 0.01, 'max_positions': 3, 'initial_capital': 10000.0,
        'check_interval': 3600, 'trading_hours': '9-17', 'time_filter': False,
        'paper_trading': False, 'notify': False, 'monitor': False,
    }
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        default_config.update(config)
    return default_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments voor live trading."""
    parser = argparse.ArgumentParser(description="Sophy4 Live Trading Tool")
    parser.add_argument("--strategy", type=str, required=True, help="Naam van de strategie")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbool")
    parser.add_argument("--timeframe", type=str, default="D1", help="Timeframe")
    parser.add_argument("--params_file", type=str, help="JSON bestand met parameters")
    parser.add_argument("--window", type=int, help="Bollinger Band window periode")
    parser.add_argument("--std_dev", type=float, help="Bollinger Band standaarddeviaties")
    parser.add_argument("--sl_method", type=str, choices=['fixed_percent', 'atr_based'], help="Stop loss methode")
    parser.add_argument("--sl_fixed_percent", type=float, help="Stop loss percentage")
    parser.add_argument("--tp_method", type=str, choices=['fixed_percent', 'atr_based'], help="Take profit methode")
    parser.add_argument("--tp_fixed_percent", type=float, help="Take profit percentage")
    parser.add_argument("--use_trailing_stop", action="store_true", help="Trailing stop gebruiken")
    parser.add_argument("--trailing_stop_percent", type=float, help="Trailing stop percentage")
    parser.add_argument("--risk_per_trade", type=float, help="Risico per trade (0-1)")
    parser.add_argument("--max_positions", type=int, help="Max aantal open posities")
    parser.add_argument("--initial_capital", type=float, help="Initieel kapitaal")
    parser.add_argument("--login", type=int, help="MT5 account nummer")
    parser.add_argument("--password", type=str, help="MT5 account wachtwoord")
    parser.add_argument("--server", type=str, help="MT5 server naam")
    parser.add_argument("--check_interval", type=int, help="Tijd tussen checks (seconden)")
    parser.add_argument("--time_filter", action="store_true", help="Handelen binnen uren")
    parser.add_argument("--trading_hours", type=str, help="Handelsuren (bijv. '9-17,22-6')")
    parser.add_argument("--paper_trading", action="store_true", help="Paper trading modus")
    parser.add_argument("--notify", action="store_true", help="Meldingen bij signalen")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring thread")
    return parser.parse_args()


def load_parameters(args: argparse.Namespace) -> Dict:
    """Laad en valideer strategieparameters."""
    config = load_config()
    parameters = {}

    if args.params_file and Path(args.params_file).exists():
        with open(args.params_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                parameters = data[0]['params']
            elif 'params' in data:
                parameters = data['params']
            elif 'parameters' in data:
                parameters = data['parameters']
            logger.info(f"Parameters geladen uit {args.params_file}")

    # Overschrijf met command-line args
    for arg, value in vars(args).items():
        if value is not None and arg in ['window', 'std_dev', 'sl_method', 'sl_fixed_percent',
                                        'tp_method', 'tp_fixed_percent', 'use_trailing_stop',
                                        'trailing_stop_percent']:
            parameters[arg] = value

    # Valideer en voeg risk_per_trade toe
    risk = args.risk_per_trade if args.risk_per_trade is not None else config['risk_per_trade']
    if not 0 < risk <= 1:
        raise ValueError("risk_per_trade moet tussen 0 en 1 liggen")
    parameters['risk_per_trade'] = risk

    return parameters


def initialize_mt5_connection(args: argparse.Namespace) -> bool:
    """Initialiseer MT5 verbinding."""
    if not mt5.initialize():
        logger.error(f"MT5 initialisatie mislukt: {mt5.last_error()}")
        return False

    if args.login and args.password and args.server:
        logger.info(f"Login op MT5 account {args.login}...")
        if not mt5.login(args.login, password=args.password, server=args.server):
            logger.error(f"MT5 login mislukt: {mt5.last_error()}")
            mt5.shutdown()
            return False
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"MT5 verbonden: Account {account_info.login} ({account_info.name})")
        else:
            logger.warning("MT5 verbonden maar geen account info")
    else:
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Bestaande MT5 verbinding: Account {account_info.login}")
        else:
            logger.warning("MT5 zonder account, paper trading modus")
    return True


def get_cached_data(symbol: str, timeframe: str, bars: int = 200) -> Optional[pd.DataFrame]:
    """Haal gecachte data op of vernieuw indien nodig."""
    global data_cache
    cache_key = (symbol, timeframe)
    refresh_interval = TIMEFRAME_INTERVALS.get(timeframe, timedelta(hours=1))
    if (cache_key not in data_cache or
            datetime.now() - data_cache[cache_key]['timestamp'] >= refresh_interval):
        df = fetch_historical_data(symbol, timeframe)

        if df is not None and not df.empty:
            data_cache[cache_key] = {'data': df, 'timestamp': datetime.now()}
    return data_cache.get(cache_key, {}).get('data')


def generate_trading_signal(strategy_name: str, parameters: Dict, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Genereer trading signalen met gecachte data."""
    df = get_cached_data(symbol, timeframe)
    if df is None or df.empty:
        logger.error(f"Geen data voor {symbol}")
        return None

    try:
        strategy = get_strategy(strategy_name, **parameters)
        entries, sl_stop, tp_stop = strategy.generate_signals(df)
        df['entries'] = entries
        df['sl_stop'] = sl_stop
        df['tp_stop'] = tp_stop
        logger.info(f"Signalen voor {symbol}: Entry {entries.iloc[-1]}, SL {sl_stop.iloc[-1]:.4f}, TP {tp_stop.iloc[-1]:.4f}")
        return df
    except Exception as e:
        logger.error(f"Fout bij signaalgeneratie: {e}")
        return None


def parse_trading_hours(trading_hours_str: str) -> List[Tuple[int, int]]:
    """Parse flexibele trading hours (bijv. '9-17,22-6')."""
    try:
        ranges = trading_hours_str.split(',')
        trading_windows = []
        for r in ranges:
            start, end = map(int, r.strip().split('-'))
            trading_windows.append((start, end))
        return trading_windows
    except Exception as e:
        logger.warning(f"Ongeldig trading_hours formaat: {e}, gebruik standaard 9-17")
        return [(9, 17)]


def is_trading_time(trading_windows: List[Tuple[int, int]]) -> bool:
    """Controleer of huidige tijd binnen trading hours valt."""
    current_hour = datetime.now().hour
    return any(start <= current_hour <= end if start <= end else current_hour >= start or current_hour <= end
               for start, end in trading_windows)


class Notifier(ABC):
    @abstractmethod
    def send(self, message: str):
        pass


class ConsoleNotifier(Notifier):
    def send(self, message: str):
        logger.info(f"NOTIFICATIE: {message}")


class EmailNotifier(Notifier):
    def send(self, message: str):
        logger.info(f"EMAIL Placeholder: {message}")  # Voeg echte e-mail logica toe


notifiers = []


def send_notification(message: str):
    """Stuur notificaties via alle geregistreerde notifiers."""
    for notifier in notifiers:
        notifier.send(message)


def cleanup(signum=None, frame=None):
    """Graceful shutdown."""
    logger.info("Cleanup uitgevoerd...")
    if not args.paper_trading:
        mt5.shutdown()
    sys.exit(0)


def main():
    """Hoofdfunctie voor live trading."""
    global args, notifiers
    args = parse_args()
    config = load_config()

    # Overschrijf config met command-line args
    for key in config:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)

    # Stel logging in
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)
    log_file = output_path / f"live_trading_{args.symbol}_{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Print configuratie
    print("\n" + "=" * 80)
    print(f"Strategie: {args.strategy}, Symbool: {args.symbol}, Timeframe: {args.timeframe}")
    print(f"Risico per trade: {config['risk_per_trade'] * 100}%, Max posities: {config['max_positions']}")
    if config['time_filter']:
        print(f"Trading uren: {args.trading_hours}")
    if config['paper_trading']:
        print("Modus: PAPER TRADING")
    print("=" * 80 + "\n")

    logger.info(f"Live trading gestart: {args.strategy} op {args.symbol} ({args.timeframe})")

    # Laad parameters
    parameters = load_parameters(args)
    if not parameters:
        logger.error("Geen geldige parameters. Stop.")
        return
    logger.info(f"Parameters: {parameters}")

    # Initialiseer MT5
    if not config['paper_trading'] and not initialize_mt5_connection(args):
        logger.error("MT5 verbinding mislukt. Stop.")
        return

    # Stel notificaties in
    if config['notify']:
        notifiers.extend([ConsoleNotifier(), EmailNotifier()])

    # Start monitoring thread
    if config['monitor']:
        trading_windows = parse_trading_hours(config['trading_hours']) if config['time_filter'] else [(0, 24)]
        monitor_thread = threading.Thread(
            target=run_monitor_loop,
            kwargs={'live': True, 'symbols': [args.symbol], 'interval': 60,
                    'time_filter': config['time_filter'], 'trading_hours': trading_windows,
                    'trailing_stop_percent': parameters.get('trailing_stop_percent', 0.015)},
            daemon=True
        )
        monitor_thread.start()
        logger.info("Monitoring thread gestart")

    # Registreer cleanup handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Hoofdlus
    last_check_time = {}
    try:
        while True:
            current_time = datetime.now()
            check_delta = TIMEFRAME_INTERVALS.get(args.timeframe, timedelta(seconds=config['check_interval']))
            should_check = (args.timeframe not in last_check_time or
                           (current_time - last_check_time[args.timeframe] >= check_delta))

            if config['time_filter']:
                trading_windows = parse_trading_hours(config['trading_hours'])
                if not is_trading_time(trading_windows):
                    logger.info(f"Buiten trading uren, wacht...")
                    time.sleep(60)
                    continue

            if should_check:
                logger.info(f"Checking signalen voor {args.symbol}...")
                df = generate_trading_signal(args.strategy, parameters, args.symbol, args.timeframe)
                if df is not None:
                    last_check_time[args.timeframe] = current_time
                    if df['entries'].iloc[-1]:
                        logger.info(f"⚠️ ENTRY SIGNAAL voor {args.symbol}")
                        if config['notify']:
                            send_notification(f"Entry signaal voor {args.symbol}")

                        if config['paper_trading']:
                            logger.info(f"PAPER TRADING: Zou trade openen met SL: {df['sl_stop'].iloc[-1]:.4f}, TP: {df['tp_stop'].iloc[-1]:.4f}")
                        else:
                            trade_result = run_live_trading(
                                df=df, symbol=args.symbol,
                                use_trailing_stop=parameters.get('use_trailing_stop', False),
                                risk_per_trade=parameters['risk_per_trade'],
                                max_positions=config['max_positions']
                            )
                            if trade_result['success'] and trade_result['trade_executed']:
                                logger.info(f"✅ Trade uitgevoerd: {args.symbol} @ {trade_result['entry_price']}")
                                if config['notify']:
                                    send_notification(f"Trade geopend: {args.symbol} @ {trade_result['entry_price']}")
                            else:
                                logger.warning(f"❌ Trade mislukt: {trade_result['message']}")
                    else:
                        logger.info(f"Geen entry signaal voor {args.symbol}")
                time.sleep(10)
            else:
                time.sleep(min(60, config['check_interval']))

    except Exception as e:
        logger.error(f"Onverwachte fout: {e}", exc_info=True)
    finally:
        cleanup()


if __name__ == "__main__":
    main()