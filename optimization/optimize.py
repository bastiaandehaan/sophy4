# optimization/optimize.py
import sys
import os
from pathlib import Path

# Voeg projectroot toe aan pythonpath
sys.path.append(str(Path(__file__).parent.parent))

import vectorbt as vbt
import pandas as pd
import numpy as np
import time
import json
import logging
import matplotlib.pyplot as plt
from itertools import product

# Nu kunnen we correct importeren
from utils.data_utils import fetch_historical_data
from strategies import get_strategy, STRATEGIES
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from config import SYMBOL, INITIAL_CAPITAL, FEES, logger, OUTPUT_DIR


def optimize_strategy(strategy_name, symbol=SYMBOL, param_ranges=None,
                      # None = gebruik strategy.get_default_params()
                      metric="sharpe_ratio",  # Primaire optimalisatie metric
                      top_n=3,  # Aantal top resultaten om terug te geven
                      initial_capital=INITIAL_CAPITAL, fees=FEES, period_days=1095,
                      ftmo_compliant_only=True, output_dir=OUTPUT_DIR, verbose=True):
    """
    Voer een grid search uit om optimale parameters te vinden voor een strategie.

    Args:
        strategy_name: Naam van de geregistreerde strategie
        symbol: Trading symbool
        param_ranges: Dictionary met parameter namen en ranges
        metric: Primaire metric om te optimaliseren
        top_n: Aantal top resultaten om terug te geven
        initial_capital: Startkapitaal voor backtests
        fees: Trading kosten percentage
        period_days: Aantal dagen voor backtest
        ftmo_compliant_only: Alleen parameter sets die FTMO-compliant zijn
        output_dir: Map om resultaten op te slaan
        verbose: Toon voortgangsinformatie

    Returns:
        dict: Optimalisatie resultaten met top parameter sets
    """
    start_time = time.time()

    # Maak output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Haal strategie class en default parameter ranges op
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Strategie '{strategy_name}' niet gevonden. Beschikbaar: {', '.join(STRATEGIES.keys())}")

    strategy_class = STRATEGIES[strategy_name]

    # Gebruik default parameter ranges indien niet opgegeven
    if param_ranges is None:
        param_ranges = strategy_class.get_default_params()

    # Haal de lijst met metrics op die deze strategie belangrijk vindt
    metrics_list = strategy_class.get_performance_metrics()
    if metric not in metrics_list:
        metrics_list.append(metric)

    # Haal historische data op
    df_original = fetch_historical_data(symbol, days=period_days)
    if df_original is None:
        logger.error(f"Kon geen historische data ophalen voor {symbol}")
        return None

    # Maak parameter combinaties
    param_names = list(param_ranges.keys())
    param_values = [
        param_ranges[name] if isinstance(param_ranges[name], list) else list(
            param_ranges[name]) for name in param_names]
    param_combinations = list(product(*param_values))

    total_combinations = len(param_combinations)
    if verbose:
        logger.info(
            f"Optimalisatie gestart met {total_combinations} parameter combinaties")
        logger.info(f"Strategie: {strategy_name}, Symbool: {symbol}")
        logger.info(f"Optimaliseren voor {metric}, top {top_n} resultaten bewaren")

    # Voorbereiden voor resultaten
    results = []
    processed = 0
    compliant_count = 0

    # Verlaag log level tijdens grid search
    original_level = logger.level
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    try:
        # Voer grid search uit
        for params in param_combinations:
            # Maak parameter dictionary
            param_dict = {name: value for name, value in zip(param_names, params)}

            # Maak strategie instantie met deze parameters
            strategy = get_strategy(strategy_name, **param_dict)

            try:
                # Valideer parameters
                strategy.validate_parameters()

                # Maak kopie van dataframe voor deze iteratie
                df = df_original.copy()

                # Genereer signalen
                entries, sl_stop, tp_stop = strategy.generate_signals(df)

                # Zorg dat het Series objecten zijn (geen views)
                entries = entries.copy() if isinstance(entries,
                                                       pd.Series) else pd.Series(
                    entries, index=df.index)
                sl_stop = sl_stop.copy() if isinstance(sl_stop,
                                                       pd.Series) else pd.Series(
                    sl_stop, index=df.index)
                tp_stop = tp_stop.copy() if isinstance(tp_stop,
                                                       pd.Series) else pd.Series(
                    tp_stop, index=df.index)
                close = df['close'].copy()  # Maak expliciet kopie van close

                # Voer backtest uit met kopieën
                pf = vbt.Portfolio.from_signals(close=close, entries=entries,
                    sl_stop=sl_stop, tp_stop=tp_stop, init_cash=initial_capital,
                    fees=fees, freq='1D')

                # Bereken performance metrics
                metrics_dict = {'total_return': pf.total_return(),
                                'sharpe_ratio': pf.sharpe_ratio(),
                                'sortino_ratio': pf.sortino_ratio(),
                                'calmar_ratio': pf.calmar_ratio(),
                                'max_drawdown': pf.max_drawdown(),
                                'win_rate': pf.trades.win_rate() if len(
                                    pf.trades) > 0 else 0,
                                'trades_count': len(pf.trades),
                                'avg_trade_duration': pf.trades[
                                    'duration'].mean() if len(pf.trades) > 0 else 0}

                # Check FTMO compliance
                compliant, profit_reached = check_ftmo_compliance(pf, metrics_dict)

                # Alleen toevoegen aan resultaten als het FTMO compliant is (indien vereist)
                if not ftmo_compliant_only or (compliant and profit_reached):
                    compliant_count += 1
                    results.append({'params': param_dict, 'metrics': metrics_dict,
                                    'ftmo_compliant': compliant,
                                    'profit_target_reached': profit_reached})

            except Exception as e:
                if verbose:
                    logger.warning(f"Fout bij testen parameters {param_dict}: {str(e)}")

            # Update voortgang
            processed += 1
            if verbose and processed % max(1, total_combinations // 20) == 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / processed) * (total_combinations - processed)
                logger.info(
                    f"Voortgang: {processed}/{total_combinations} ({processed / total_combinations * 100:.1f}%) | "
                    f"Verstreken: {elapsed:.1f}s | Resterend: {remaining:.1f}s | "
                    f"FTMO compliant: {compliant_count}")

        # Sorteer resultaten op de primaire metric
        # Let op: voor drawdown is lager beter, dus sorteren we anders
        reverse = metric != 'max_drawdown'
        sorted_results = sorted(results, key=lambda x: x['metrics'][metric],
                                reverse=reverse)

        # Neem top N resultaten
        top_results = sorted_results[:top_n]

        # Genereer gedetailleerd rapport voor top resultaten
        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info(f"OPTIMALISATIE SAMENVATTING VOOR {strategy_name}")
            logger.info("=" * 60)
            logger.info(f"Totaal geteste combinaties: {processed}/{total_combinations}")
            logger.info(
                f"FTMO compliant combinaties: {compliant_count} ({compliant_count / processed * 100:.1f}%)")
            logger.info(f"Totale tijd: {time.time() - start_time:.1f} seconden")

            # Toon top resultaten
            logger.info("\nTOP RESULTATEN:")
            for i, result in enumerate(top_results):
                logger.info(f"\n#{i + 1}: {metric} = {result['metrics'][metric]:.4f}")
                for param_name, param_value in result['params'].items():
                    logger.info(f"  {param_name}: {param_value}")
                logger.info("  --- Performance ---")
                logger.info(f"  Return: {result['metrics']['total_return']:.2%}")
                logger.info(f"  Sharpe: {result['metrics']['sharpe_ratio']:.2f}")
                logger.info(f"  Sortino: {result['metrics']['sortino_ratio']:.2f}")
                logger.info(f"  Calmar: {result['metrics']['calmar_ratio']:.2f}")
                logger.info(f"  Drawdown: {result['metrics']['max_drawdown']:.2%}")
                logger.info(f"  Win Rate: {result['metrics']['win_rate']:.2%}")
                logger.info(f"  Trades: {result['metrics']['trades_count']}")

        # Sla resultaten op in bestand
        results_file = output_path / f"{strategy_name}_{symbol}_optimization.json"
        with open(results_file, 'w') as f:
            json.dump({'strategy': strategy_name, 'symbol': symbol,
                       'total_combinations': total_combinations,
                       'ftmo_compliant_count': compliant_count,
                       'optimization_time': time.time() - start_time, 'top_results': [
                    {'params': result['params'],
                     'metrics': {k: float(v) for k, v in result['metrics'].items()}} for
                    result in top_results]}, f, indent=2)

        if verbose:
            logger.info(f"\nResultaten opgeslagen in {results_file}")

        # Verifieer het beste resultaat met een aparte backtest
        if top_results:
            best_params = top_results[0]['params']
            if verbose:
                logger.info("\nVerifiëren van beste resultaat met aparte backtest...")

            # Maak een nieuwe kopie van de data voor verificatie
            df_verify = df_original.copy()

            strategy = get_strategy(strategy_name, **best_params)
            entries, sl_stop, tp_stop = strategy.generate_signals(df_verify)

            # Converteer en kopieer data voor verificatie
            entries = entries.copy() if isinstance(entries, pd.Series) else pd.Series(
                entries, index=df_verify.index)
            sl_stop = sl_stop.copy() if isinstance(sl_stop, pd.Series) else pd.Series(
                sl_stop, index=df_verify.index)
            tp_stop = tp_stop.copy() if isinstance(tp_stop, pd.Series) else pd.Series(
                tp_stop, index=df_verify.index)
            close = df_verify['close'].copy()

            pf = vbt.Portfolio.from_signals(close=close, entries=entries,
                sl_stop=sl_stop, tp_stop=tp_stop, init_cash=initial_capital, fees=fees,
                freq='1D')

            # Maak equity curve plot
            plt.figure(figsize=(12, 6))
            pf.plot()
            plt.title(f"{strategy_name} Equity Curve (Geoptimaliseerde Parameters)")
            plt.savefig(output_path / f"{strategy_name}_{symbol}_optimized_equity.png")
            plt.close()

            if verbose:
                logger.info(f"Verificatie backtest voltooid. Equity curve opgeslagen.")

    finally:
        # Herstel log level
        logger.setLevel(original_level)

    return {'top_results': top_results, 'total_combinations': total_combinations,
            'compliant_count': compliant_count, 'time_taken': time.time() - start_time}


def validate_best_parameters(strategy_name, params, symbol=SYMBOL,
                             output_dir=OUTPUT_DIR):
    """
    Voer een gedetailleerde backtest uit met de beste parameters om de resultaten te valideren.

    Args:
        strategy_name: Naam van de strategie
        params: Dictionary met parameters
        symbol: Trading symbool
        output_dir: Map om resultaten op te slaan

    Returns:
        dict: Validatie resultaten
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Haal historische data op
    df_original = fetch_historical_data(symbol)
    if df_original is None:
        logger.error(f"Kon geen historische data ophalen voor {symbol}")
        return None

    # Maak kopie voor deze validatie
    df = df_original.copy()

    # Maak strategie instantie met deze parameters
    strategy = get_strategy(strategy_name, **params)

    # Genereer signalen
    entries, sl_stop, tp_stop = strategy.generate_signals(df)

    # Zorg dat het Series objecten zijn (geen views)
    entries = entries.copy() if isinstance(entries, pd.Series) else pd.Series(entries,
                                                                              index=df.index)
    sl_stop = sl_stop.copy() if isinstance(sl_stop, pd.Series) else pd.Series(sl_stop,
                                                                              index=df.index)
    tp_stop = tp_stop.copy() if isinstance(tp_stop, pd.Series) else pd.Series(tp_stop,
                                                                              index=df.index)
    close = df['close'].copy()

    # Voer backtest uit
    pf = vbt.Portfolio.from_signals(close=close, entries=entries, sl_stop=sl_stop,
        tp_stop=tp_stop, init_cash=INITIAL_CAPITAL, fees=FEES, freq='1D')

    # Bereken metrics
    metrics = {'total_return': pf.total_return(), 'sharpe_ratio': pf.sharpe_ratio(),
               'sortino_ratio': pf.sortino_ratio(), 'calmar_ratio': pf.calmar_ratio(),
               'max_drawdown': pf.max_drawdown(),
               'win_rate': pf.trades.win_rate() if len(pf.trades) > 0 else 0,
               'trades_count': len(pf.trades)}

    # Check FTMO compliance
    compliant, profit_reached = check_ftmo_compliance(pf, metrics)

    # Maak gedetailleerd rapport
    logger.info("\n" + "=" * 60)
    logger.info(f"VALIDATIE RESULTATEN VOOR {strategy_name}")
    logger.info("=" * 60)
    logger.info("\nParameters:")
    for param_name, param_value in params.items():
        logger.info(f"  {param_name}: {param_value}")

    logger.info("\nPerformance Metrics:")
    logger.info(f"  Return: {metrics['total_return']:.2%}")
    logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Sortino: {metrics['sortino_ratio']:.2f}")
    logger.info(f"  Calmar: {metrics['calmar_ratio']:.2f}")
    logger.info(f"  Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"  Trades: {metrics['trades_count']}")

    logger.info("\nFTMO Compliance:")
    logger.info(f"  Compliant: {'JA' if compliant else 'NEE'}")
    logger.info(f"  Winstdoelstelling Behaald: {'JA' if profit_reached else 'NEE'}")

    # Maak plots
    # 1. Equity curve
    plt.figure(figsize=(12, 6))
    pf.plot()
    plt.title(f"{strategy_name} Equity Curve (Gevalideerde Parameters)")
    plt.savefig(output_path / f"{strategy_name}_{symbol}_validated_equity.png")
    plt.close()

    # 2. Drawdown plot
    plt.figure(figsize=(12, 6))
    pf.drawdown().plot()
    plt.title(f"{strategy_name} Drawdowns")
    plt.savefig(output_path / f"{strategy_name}_{symbol}_drawdowns.png")
    plt.close()

    # 3. Maandelijkse returns heatmap (indien beschikbaar)
    try:
        plt.figure(figsize=(12, 8))
        monthly_returns = pf.returns().resample('M').sum()
        monthly_returns = monthly_returns.to_frame()
        plt.imshow([monthly_returns.values.flatten()], cmap='RdYlGn', aspect='auto')
        plt.title(f"{strategy_name} Maandelijkse Returns")
        plt.savefig(output_path / f"{strategy_name}_{symbol}_monthly_returns.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Kon geen maandelijkse returns heatmap maken: {str(e)}")

    logger.info(f"\nValidatie plots opgeslagen in {output_path}")

    # Sla gedetailleerde trade lijst op
    try:
        trades_file = output_path / f"{strategy_name}_{symbol}_trades.csv"
        pf.trades.records_readable.to_csv(trades_file)
        logger.info(f"Trade lijst opgeslagen in {trades_file}")
    except Exception as e:
        logger.warning(f"Kon trade lijst niet opslaan: {str(e)}")

    return {'metrics': metrics, 'ftmo_compliant': compliant,
            'profit_target_reached': profit_reached, 'portfolio': pf}


def main():
    """Command-line interface voor optimalisatie."""
    import argparse

    parser = argparse.ArgumentParser(description="Sophy4 Strategie Optimizer")

    parser.add_argument("--strategy", type=str, required=True,
                        help="Naam van de strategie")
    parser.add_argument("--symbol", type=str, default=SYMBOL,
                        help=f"Trading symbool (default: {SYMBOL})")
    parser.add_argument("--metric", type=str, default="sharpe_ratio",
                        choices=["sharpe_ratio", "sortino_ratio", "calmar_ratio",
                                 "total_return", "max_drawdown", "win_rate"],
                        help="Metric om te optimaliseren (default: sharpe_ratio)")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Aantal top resultaten (default: 3)")
    parser.add_argument("--ftmo_only", action="store_true",
                        help="Alleen FTMO compliant parameter sets")
    parser.add_argument("--apply_best", action="store_true",
                        help="Voer gedetailleerde validatie uit van beste parameter set")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")

    args = parser.parse_args()

    # Voer optimalisatie uit
    results = optimize_strategy(strategy_name=args.strategy, symbol=args.symbol,
                                metric=args.metric, top_n=args.top_n,
                                ftmo_compliant_only=args.ftmo_only,
                                output_dir=args.output_dir, verbose=True)

    if results and args.apply_best and results['top_results']:
        # Valideer de beste parameter set
        best_params = results['top_results'][0]['params']
        validate_best_parameters(strategy_name=args.strategy, params=best_params,
                                 symbol=args.symbol, output_dir=args.output_dir)


if __name__ == "__main__":
    main()