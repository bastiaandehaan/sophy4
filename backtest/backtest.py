# backtest/backtest.py
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import vectorbt as vbt
from utils.plotting import create_visualizations

from backtest.data_loader import fetch_historical_data
from config import INITIAL_CAPITAL, FEES, OUTPUT_DIR, logger
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from risk.risk_management import RiskManager
from strategies import get_strategy


def _calculate_stop(portfolio_kwargs: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    """
    Bereken de stop losses op basis van de strategie parameters.
    Wijzigt portfolio_kwargs in-place.
    """
    # Add trailing stop if specified
    if parameters.get('use_trailing_stop', False):
        trailing_stop = parameters.get('trailing_stop_percent', 0.02)
        portfolio_kwargs['sl_stop'] = portfolio_kwargs['sl_stop'].fillna(trailing_stop)
        portfolio_kwargs['sl_trail'] = True

        # Als er een activatiepunt is gespecificeerd
        trail_start = parameters.get('trailing_activation_percent', 0)
        if trail_start > 0:
            portfolio_kwargs['sl_trail_start'] = trail_start
    else:
        # Normale stop zonder trailing
        portfolio_kwargs['sl_trail'] = False


def calculate_metrics(pf: vbt.Portfolio) -> Dict[str, Any]:
    """
    Berekent de prestatiemetrieken van een portfolio.
    """
    print("DEBUG: Metriek berekening gestart")
    try:
        metrics = {}

        # Rendement berekenen (als percentage)
        metrics['total_return'] = pf.total_return()

        # Sharpe ratio
        if not np.isnan(pf.sharpe_ratio()):
            metrics['sharpe_ratio'] = pf.sharpe_ratio()
        else:
            metrics['sharpe_ratio'] = 0.0

        # Sortino ratio
        sortino = pf.sortino_ratio()
        metrics['sortino_ratio'] = 0.0 if np.isnan(sortino) else sortino

        # Drawdown
        max_dd = pf.max_drawdown()
        metrics['max_drawdown'] = 0.0 if np.isnan(max_dd) else max_dd

        # CAGR (Compound Annual Growth Rate)
        if pf.annualized_return() is not None and not np.isnan(pf.annualized_return()):
            metrics['cagr'] = pf.annualized_return()
        else:
            metrics['cagr'] = 0.0

        # Calmar ratio (verhouding rendement/max drawdown)
        if metrics['cagr'] > 0 and metrics['max_drawdown'] < 0:
            metrics['calmar_ratio'] = abs(metrics['cagr'] / metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0.0

        # Win rate
        if len(pf.trades) > 0:
            metrics['win_rate'] = pf.trades.win_rate
            metrics['trades_count'] = len(pf.trades)
            metrics['avg_winning_trade'] = pf.trades.winning.pnl.mean() if len(pf.trades.winning) > 0 else 0
            metrics['avg_losing_trade'] = pf.trades.losing.pnl.mean() if len(pf.trades.losing) > 0 else 0

            # Profit factor (som van winsten / som van verliezen)
            total_win = pf.trades.winning.pnl.sum() if len(pf.trades.winning) > 0 else 0
            total_loss = abs(pf.trades.losing.pnl.sum()) if len(pf.trades.losing) > 0 else 0
            metrics['profit_factor'] = total_win / total_loss if total_loss > 0 else float('inf')
        else:
            metrics['win_rate'] = 0.0
            metrics['trades_count'] = 0
            metrics['avg_winning_trade'] = 0.0
            metrics['avg_losing_trade'] = 0.0
            metrics['profit_factor'] = 0.0

        print(f"DEBUG: Metrics berekend: {metrics}")
        return metrics
    except Exception as e:
        print(f"FOUT bij berekenen metrics: {str(e)}")
        logger.error(f"Fout bij berekenen metrics: {str(e)}", exc_info=True)
        return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'trades_count': 0}


def calculate_income_metrics(pf: vbt.Portfolio, metrics: Dict[str, Any], initial_capital: float) -> Dict[str, Any]:
    """
    Berekent de absolute inkomsten metrics op basis van een portfolio.
    """
    print("DEBUG: Inkomensmetrieken berekening gestart")
    try:
        income_metrics = {}

        # Absolute winst
        income_metrics['absolute_profit'] = initial_capital * metrics['total_return']

        # Gemiddelde absolute winst per trade
        if metrics['trades_count'] > 0:
            income_metrics['avg_profit_per_trade'] = income_metrics['absolute_profit'] / metrics['trades_count']
        else:
            income_metrics['avg_profit_per_trade'] = 0.0

        # Gemiddelde winst per maand
        days = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days
        if days > 0:
            months = days / 30.0  # Approximation
            income_metrics['avg_monthly_profit'] = income_metrics['absolute_profit'] / months
        else:
            income_metrics['avg_monthly_profit'] = 0.0

        print(f"DEBUG: Inkomensmetrieken berekend: {income_metrics}")
        return income_metrics
    except Exception as e:
        print(f"FOUT bij berekenen inkomensmetrieken: {str(e)}")
        logger.error(f"Fout bij berekenen inkomensmetrieken: {str(e)}", exc_info=True)
        return {'absolute_profit': 0.0, 'avg_profit_per_trade': 0.0, 'avg_monthly_profit': 0.0}


def run_extended_backtest(strategy_name: str, parameters: Dict[str, Any], symbol: str,
                          timeframe: Optional[Union[str, int]] = None,
                          period_days: int = 1095,
                          initial_capital: float = INITIAL_CAPITAL,
                          end_date: Optional[datetime] = None) -> Tuple[
    Optional[vbt.Portfolio], Dict[str, Any]]:
    """
    Voer een uitgebreide backtest uit met metrics en visualisaties.

    Args:
        strategy_name: Naam van de strategie.
        parameters: Strategieparameters.
        symbol: Ticker van het instrument.
        timeframe: Timeframe (bijv. 'D1' of mt5.TIMEFRAME_D1).
        period_days: Aantal dagen historische data.
        initial_capital: Startkapitaal.
        end_date: Einddatum voor de test (standaard: nu).

    Returns:
        Tuple van (Portfolio-object of None, dictionary met metrics).
    """
    print(f"\nDEBUG: Start backtest voor {strategy_name} op {symbol}")
    print(f"DEBUG: Parameters: {parameters}")

    output_path: Path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)
    timestamp: str = datetime.now().strftime("%Y%m%d")

    logger.info(f"Start backtest: {strategy_name} op {symbol} met {period_days} dagen data")
    logger.info(f"Timeframe: {timeframe}, Parameters: {parameters}")

    try:
        # Fetch data and validate
        print(f"DEBUG: Data ophalen voor {symbol}, timeframe={timeframe}, days={period_days}")
        df: Optional[pd.DataFrame] = fetch_historical_data(symbol, timeframe=timeframe, days=period_days,
                                                           end_date=end_date)

        if df is None or df.empty:
            print(f"FOUT: Geen geldige data voor {symbol}")
            logger.error(f"Geen geldige data voor {symbol}")
            return None, {}

        print(f"DEBUG: Data opgehaald, aantal rijen: {len(df)}")
        print(f"DEBUG: Eerste rij: {df.iloc[0]}")
        print(f"DEBUG: Laatste rij: {df.iloc[-1]}")

        # Initialiseer RiskManager
        print("DEBUG: RiskManager initialiseren")
        risk_manager: RiskManager = RiskManager(
            confidence_level=parameters.get('confidence_level', 0.95),
            max_risk=parameters.get('risk_per_trade', 0.01)
        )

        # Bereken rendementen voor VaR
        returns: pd.Series = df['close'].pct_change().dropna()

        # Haal dynamische pip-waarde op via RiskManager
        symbol_info = risk_manager.get_symbol_info(symbol)
        if symbol_info:
            pip_value = symbol_info["pip_value"]
            logger.info(f"Dynamische pip-waarde voor {symbol}: {pip_value}")
        else:
            pip_value = 10.0  # Fallback waarde
            logger.warning(f"Kon pip-waarde niet ophalen voor {symbol}, fallback naar {pip_value}")

        # Gebruik historische spread uit de DataFrame (in punten), converteer naar prijs-eenheden
        if 'spread' in df.columns and not df['spread'].empty:
            avg_spread = df['spread'].mean() * pip_value  # Spread in prijs-eenheden
            logger.info(f"Gemiddelde spread voor {symbol}: {avg_spread} (gebaseerd op historische data)")
        else:
            avg_spread = 0.0
            logger.warning(f"Geen spread-data beschikbaar in DataFrame voor {symbol}, spread ingesteld op 0")

        # Bereken positiegrootte met dynamische pip-waarde en historische spread
        size: float = risk_manager.calculate_position_size(
            capital=initial_capital,
            returns=returns,
            pip_value=pip_value,
            symbol=symbol
        )
        logger.info(f"VaR-gebaseerde positiegrootte voor {symbol}: {size}")
        print(f"DEBUG: Positiegrootte berekend: {size}")

        # Generate signals
        print(f"DEBUG: Strategie initialiseren: {strategy_name}")
        strategy = get_strategy(strategy_name, **parameters)
        print("DEBUG: Strategie geïnitialiseerd, genereren signalen...")

        entries, sl_stop, tp_stop = strategy.generate_signals(df)

        print(f"DEBUG: Signalen gegenereerd:")
        print(f"DEBUG: Entries type: {type(entries)}, lengte: {len(entries)}, aantal True: {entries.sum()}")
        print(f"DEBUG: SL type: {type(sl_stop)}, lengte: {len(sl_stop)}, gemiddelde: {sl_stop.mean()}")
        print(f"DEBUG: TP type: {type(tp_stop)}, lengte: {len(tp_stop)}, gemiddelde: {tp_stop.mean()}")

        if not all(len(x) == len(df) for x in [entries, sl_stop, tp_stop]):
            print("FOUT: Signalen hebben inconsistente lengtes")
            print(f"DEBUG: Lengtes - df: {len(df)}, entries: {len(entries)}, sl_stop: {len(sl_stop)}, tp_stop: {len(tp_stop)}")
            raise ValueError("Signalen hebben inconsistente lengtes")

        # Map timeframe to freq for vectorbt (use lowercase 'h' to avoid FutureWarning)
        timeframe_to_freq = {
            'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
            'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w', 'MN1': '1M'
        }
        freq = timeframe_to_freq.get(str(timeframe), '1d')  # Default to '1d' if timeframe not found
        logger.info(f"Using freq: {freq} for timeframe: {timeframe}")
        print(f"DEBUG: Frequentie voor vectorbt: {freq}")

        # Create portfolio
        print("DEBUG: Portfolio voorbereiden")
        portfolio_kwargs: Dict[str, Any] = {
            'close': df['close'],
            'entries': entries,
            'sl_stop': sl_stop,
            'tp_stop': tp_stop,
            'init_cash': initial_capital,
            'fees': FEES,
            'freq': freq,
            'size': size,
        }

        _calculate_stop(portfolio_kwargs, parameters)
        print("DEBUG: Portfolio aanmaken met vectorbt")

        # Capture meer informatie over indexen om debug te vergemakkelijken
        print(f"DEBUG: DataFrame index type: {type(df.index)}")
        print(f"DEBUG: Entries index type: {type(entries.index)}")
        print(f"DEBUG: SL index type: {type(sl_stop.index)}")
        print(f"DEBUG: TP index type: {type(tp_stop.index)}")

        # Check voor index gelijkheid
        if not (df.index.equals(entries.index) and df.index.equals(sl_stop.index) and df.index.equals(tp_stop.index)):
            print("WAARSCHUWING: Indexes komen niet overeen, aanpassen...")
            # Probeer index aan te passen als ze niet overeenkomen
            entries = pd.Series(entries.values, index=df.index)
            sl_stop = pd.Series(sl_stop.values, index=df.index)
            tp_stop = pd.Series(tp_stop.values, index=df.index)

        try:
            pf: vbt.Portfolio = vbt.Portfolio.from_signals(**portfolio_kwargs)
            print("DEBUG: Portfolio aangemaakt")
        except Exception as port_error:
            print(f"FOUT bij aanmaken portfolio: {str(port_error)}")
            logger.error(f"Portfolio aanmaken mislukt: {str(port_error)}", exc_info=True)

            # Probeer met simpelere parameters als reguliere aanmaak mislukt
            try:
                print("DEBUG: Proberen met vereenvoudigde parameters...")
                simple_portfolio_kwargs = {
                    'close': df['close'],
                    'entries': entries,
                    'init_cash': initial_capital,
                    'fees': FEES,
                    'freq': freq
                }
                pf = vbt.Portfolio.from_signals(**simple_portfolio_kwargs)
                print("DEBUG: Vereenvoudigd portfolio aangemaakt")
            except Exception as e:
                print(f"FOUT bij aanmaken vereenvoudigd portfolio: {str(e)}")
                return None, {}

        # Check drawdown
        print("DEBUG: Drawdown controleren")
        current_value: float = pf.value().iloc[-1]
        max_value: float = pf.value().max()
        if risk_manager.monitor_drawdown(current_value, max_value):
            print("WAARSCHUWING: Maximale drawdown overschreden tijdens backtest!")
            logger.warning("Maximale drawdown overschreden tijdens backtest!")

        # Calculate metrics
        print("DEBUG: Metrieken berekenen")
        metrics: Dict[str, float] = calculate_metrics(pf)
        income_metrics: Dict[str, Any] = calculate_income_metrics(pf, metrics, initial_capital)

        print("DEBUG: FTMO compliance controleren")
        compliant, profit_target = check_ftmo_compliance(pf, metrics)
        all_metrics: Dict[str, Any] = {**metrics, **income_metrics, 'ftmo_compliant': compliant,
                                       'profit_target_reached': profit_target}

        # Log results
        logger.info(f"\n===== BACKTEST RESULTATEN VOOR {strategy_name} OP {symbol} =====")
        logger.info(f"Totaal rendement: {metrics['total_return']:.2%}")
        logger.info(f"Sharpe: {metrics['sharpe_ratio']:.2f}, Max drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Win rate: {metrics['win_rate']:.2%}, Trades: {metrics['trades_count']}")
        logger.info(f"FTMO compliant: {'JA' if compliant else 'NEE'}, Profit target bereikt: {'JA' if profit_target else 'NEE'}")

        # Create visualizations
        print("DEBUG: Visualisaties aanmaken")
        try:
            create_visualizations(pf, strategy_name, symbol, timeframe, output_path, timestamp)
            print("DEBUG: Visualisaties aangemaakt")
        except Exception as e:
            print(f"WAARSCHUWING: Visualisaties konden niet worden gemaakt: {str(e)}")
            logger.warning(f"Visualisaties konden niet worden gemaakt: {str(e)}")

        # Save trades and results
        print("DEBUG: Resultaten opslaan")
        timeframe_str: str = f"_{timeframe}" if timeframe else ""
        if len(pf.trades) > 0:
            try:
                pf.trades.records_readable.to_csv(
                    output_path / f"{strategy_name}_{symbol}{timeframe_str}_trades_{timestamp}.csv")
                logger.info("Trade gegevens opgeslagen")
                print("DEBUG: Trade gegevens opgeslagen")
            except Exception as e:
                print(f"WAARSCHUWING: Kon trades niet opslaan: {str(e)}")
                logger.warning(f"Kon trades niet opslaan: {str(e)}")

        try:
            with open(output_path / f"{strategy_name}_{symbol}{timeframe_str}_results_{timestamp}.json", 'w') as f:
                json_metrics = {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
                                for k, v in all_metrics.items()}
                json.dump({'strategy': strategy_name, 'symbol': symbol, 'timeframe': str(timeframe),
                           'metrics': json_metrics}, f, indent=2)
                logger.info("Resultaten opgeslagen als JSON")
                print("DEBUG: Resultaten opgeslagen als JSON")
        except Exception as e:
            print(f"WAARSCHUWING: Kon JSON resultaten niet opslaan: {str(e)}")
            logger.warning(f"Kon JSON resultaten niet opslaan: {str(e)}")

        # Print resultaten altijd naar console, zelfs als er iets mis gaat in de main functie
        print("\n=== BACKTEST RESULTATEN ===")
        print(f"Totaal rendement: {metrics['total_return']:.2%}")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win rate: {metrics['win_rate']:.2%}")
        print(f"Aantal trades: {metrics['trades_count']}")
        print(f"FTMO compliant: {'JA' if compliant else 'NEE'}")

        return pf, all_metrics

    except ValueError as ve:
        print(f"FOUT: Validatiefout in backtest voor {symbol}: {str(ve)}")
        logger.error(f"Validatiefout in backtest voor {symbol}: {str(ve)}")
        return None, {}
    except Exception as e:
        print(f"FOUT: Backtest mislukt voor {symbol}: {str(e)}")
        logger.error(f"Backtest mislukt voor {symbol}: {str(e)}", exc_info=True)

        # Probeer nog steeds basisinformatie te tonen als er een fout optreedt
        print("\nFout bij backtest uitvoering. Controleer logs voor details.")
        return None, {}


def run_monte_carlo_simulation(pf: vbt.Portfolio, n_simulations: int = 1000, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Voert een Monte Carlo simulatie uit op een backtest portfolio.

    Args:
        pf: Portfolio object van een backtest.
        n_simulations: Aantal simulaties.
        confidence_level: Betrouwbaarheidsniveau (0-1).

    Returns:
        Dictionary met simulatieresultaten.
    """
    logger.info(f"Start Monte Carlo simulatie met {n_simulations} simulaties")

    if len(pf.trades) < 10:
        logger.warning("Te weinig trades voor betrouwbare Monte Carlo simulatie")
        return {
            'return_mean': 0.0,
            'return_95ci_lower': 0.0,
            'return_95ci_upper': 0.0,
            'drawdown_mean': 0.0,
            'drawdown_95ci': 0.0,
            'profit_probability': 0.0
        }

    try:
        # Extract trade returns (PnL percentage of size)
        trade_returns = pf.trades.pnl / pf.trades.size

        # Run simulations with resampling
        np.random.seed(42)  # For reproducibility
        sim_returns = []
        sim_drawdowns = []

        for _ in range(n_simulations):
            # Resample with replacement
            sampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)

            # Calculate cumulative return
            cum_return = (1 + sampled_returns).prod() - 1
            sim_returns.append(cum_return)

            # Calculate max drawdown
            cum_prod = (1 + sampled_returns).cumprod()
            running_max = np.maximum.accumulate(cum_prod)
            drawdown = (cum_prod / running_max) - 1
            sim_drawdowns.append(drawdown.min())

        # Calculate metrics
        return_mean = np.mean(sim_returns)
        return_ci_lower = np.percentile(sim_returns, (1 - confidence_level) * 100 / 2)
        return_ci_upper = np.percentile(sim_returns, 100 - (1 - confidence_level) * 100 / 2)
        drawdown_mean = np.mean(sim_drawdowns)
        drawdown_ci = np.percentile(sim_drawdowns, (1 - confidence_level) * 100)
        profit_probability = np.mean(np.array(sim_returns) > 0)

        results = {
            'return_mean': return_mean,
            'return_95ci_lower': return_ci_lower,
            'return_95ci_upper': return_ci_upper,
            'drawdown_mean': drawdown_mean,
            'drawdown_95ci': drawdown_ci,
            'profit_probability': profit_probability
        }
        logger.info(f"Monte Carlo resultaten: Gem. rendement: {return_mean:.2%}, Winstkans: {profit_probability:.2%}")
        return results

    except Exception as e:
        logger.error(f"Fout in Monte Carlo simulatie: {str(e)}", exc_info=True)
        return {
            'return_mean': 0.0,
            'return_95ci_lower': 0.0,
            'return_95ci_upper': 0.0,
            'drawdown_mean': 0.0,
            'drawdown_95ci': 0.0,
            'profit_probability': 0.0
        }


def run_walk_forward_test(strategy_name: str, parameters: Dict[str, Any], symbol: str,
                          timeframe: Optional[Union[str, int]] = None, total_days: int = 1095,
                          test_percent: float = 0.3, windows: int = 3) -> Dict[str, Any]:
    """
    Voert een walk-forward test uit met meerdere vensters.

    Args:
        strategy_name: Naam van de strategie.
        parameters: Strategieparameters.
        symbol: Ticker symbool.
        timeframe: Timeframe.
        total_days: Totaal aantal dagen voor de hele test.
        test_percent: Percentage voor test set per venster.
        windows: Aantal vensters om te gebruiken.

    Returns:
        Dictionary met resultaten.
    """
    logger.info(f"Start walk-forward test met {windows} vensters")

    # Bepaal de grootte van elk venster
    window_days = total_days // windows

    end_date = datetime.now()

    train_returns = []
    test_returns = []
    train_sharpes = []
    test_sharpes = []
    window_results = []

    for i in range(windows):
        # Bereken de start- en einddatum voor dit venster
        window_end = end_date - timedelta(days=i * window_days)
        window_start = window_end - timedelta(days=window_days)

        # Bereken de verdeling van train/test binnen het venster
        test_days = int(window_days * test_percent)
        train_days = window_days - test_days

        train_end = window_end - timedelta(days=test_days)

        logger.info(f"Venster {i+1}: Train van {window_start} tot {train_end}, "
                    f"Test van {train_end} tot {window_end}")

        # Voer backtest uit op train set
        train_pf, train_metrics = run_extended_backtest(
            strategy_name=strategy_name,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            period_days=train_days,
            end_date=train_end
        )

        # Voer backtest uit op test set
        test_pf, test_metrics = run_extended_backtest(
            strategy_name=strategy_name,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            period_days=test_days,
            end_date=window_end
        )

        # Sla resultaten op
        window_result = {
            'window': i+1,
            'train_period': f"{window_start.strftime('%Y-%m-%d')} tot {train_end.strftime('%Y-%m-%d')}",
            'test_period': f"{train_end.strftime('%Y-%m-%d')} tot {window_end.strftime('%Y-%m-%d')}",
            'train_return': train_metrics.get('total_return', 0),
            'test_return': test_metrics.get('total_return', 0),
            'train_sharpe': train_metrics.get('sharpe_ratio', 0),
            'test_sharpe': test_metrics.get('sharpe_ratio', 0),
        }

        window_results.append(window_result)
        train_returns.append(train_metrics.get('total_return', 0))
        test_returns.append(test_metrics.get('total_return', 0))
        train_sharpes.append(train_metrics.get('sharpe_ratio', 0))
        test_sharpes.append(test_metrics.get('sharpe_ratio', 0))

        logger.info(f"Venster {i+1} resultaten - Train: {train_metrics.get('total_return', 0):.2%}, "
                    f"Test: {test_metrics.get('total_return', 0):.2%}")

    # Bereken gemiddelde metriek en robustheid scores
    avg_train_return = sum(train_returns) / len(train_returns) if train_returns else 0
    avg_test_return = sum(test_returns) / len(test_returns) if test_returns else 0
    avg_train_sharpe = sum(train_sharpes) / len(train_sharpes) if train_sharpes else 0
    avg_test_sharpe = sum(test_sharpes) / len(test_sharpes) if test_sharpes else 0

    # Bereken robustheid
    return_ratio = avg_test_return / avg_train_return if avg_train_return > 0 else 0
    sharpe_ratio = avg_test_sharpe / avg_train_sharpe if avg_train_sharpe > 0 else 0

    # Een strategie is robuust als zowel de return_ratio als de sharpe_ratio > 0.7
    is_robust = return_ratio > 0.7 and sharpe_ratio > 0.7

    results = {
        'windows_tested': windows,
        'avg_train_return': avg_train_return,
        'avg_test_return': avg_test_return,
        'avg_train_sharpe': avg_train_sharpe,
        'avg_test_sharpe': avg_test_sharpe,
        'window_results': window_results,
        'robustness': {
            'return_ratio': return_ratio,
            'sharpe_ratio': sharpe_ratio,
            'is_robust': is_robust
        }
    }

    logger.info(f"Walk-forward resultaten: Robustheid score: {return_ratio:.2f}, Is robuust: {is_robust}")
    return results