# Sophy4/main.py
import argparse

from backtest.backtest import run_backtest
from config import SYMBOL, logger
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from live.live_trading import run_live_trading
from monitor.monitor import monitor_performance
from risk.risk_management import calculate_position_size
from strategies import STRATEGIES
from utils.data_utils import fetch_historical_data, fetch_live_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sophy4 Trading Framework")

    parser.add_argument("--mode", type=str, choices=["backtest", "live"],
                        default="backtest", help="Trading mode")
    parser.add_argument("--symbol", type=str, default=SYMBOL,
                        help=f"Trading symbol (default: {SYMBOL})")
    parser.add_argument("--strategy", type=str, default="BollongStrategy",
                        help="Strategie om te gebruiken")

    # Voeg algemene strategie parameters toe
    parser.add_argument("--window", type=int, help="Window periode")
    parser.add_argument("--std_dev", type=float, help="Standaarddeviatie")
    parser.add_argument("--sl_atr_mult", type=float, help="Stop-loss ATR multiplier")
    parser.add_argument("--tp_atr_mult", type=float, help="Take-profit ATR multiplier")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Controleer of de gevraagde strategie bestaat
    if args.strategy not in STRATEGIES:
        available_strategies = ", ".join(STRATEGIES.keys())
        logger.error(f"Strategie '{args.strategy}' niet gevonden. "
                     f"Beschikbare strategieën: {available_strategies}")
        return

    # Maak een strategie-instantie
    strategy_class = STRATEGIES[args.strategy]

    # Verzamel de parameters van command line in een dict en filter None-waarden
    strategy_params = {}
    for param in ["window", "std_dev", "sl_atr_mult", "tp_atr_mult"]:
        value = getattr(args, param, None)
        if value is not None:
            strategy_params[param] = value

    # Maak een instantie van de geselecteerde strategie met de parameters
    strategy = strategy_class(**strategy_params)
    logger.info(f"Gebruik strategie: {args.strategy} met parameters: {strategy_params}")

    if args.mode == "backtest":
        df = fetch_historical_data(args.symbol)
        if df is None:
            return

        # Genereer signalen met de strategie
        entries, sl_stop, tp_stop = strategy.generate_signals(df)
        df['entries'] = entries
        df['sl_stop'] = sl_stop
        df['tp_stop'] = tp_stop

        # Run backtest
        pf, metrics = run_backtest(df, args.symbol)
        compliant, profit_reached = check_ftmo_compliance(pf, metrics)
        monitor_performance(pf)

    elif args.mode == "live":
        df = fetch_live_data(args.symbol)
        if df is None:
            return

        # Genereer signalen met de strategie
        entries, sl_stop, tp_stop = strategy.generate_signals(df)
        df['entries'] = entries
        df['sl_stop'] = sl_stop
        df['tp_stop'] = tp_stop

        # Live trading uitvoeren
        run_live_trading(df, args.symbol)
        size = calculate_position_size(10000, df['close'].iloc[-1],
                                       df['sl_stop'].iloc[-1])
        monitor_performance(live=True)


if __name__ == "__main__":
    main()