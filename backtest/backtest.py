# backtest/backtest.py
import calendar
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from matplotlib.colors import LinearSegmentedColormap

from config import INITIAL_CAPITAL, FEES, OUTPUT_DIR, logger
from ftmo_compliance.ftmo_check import check_ftmo_compliance
from risk.risk_management import RiskManager
from strategies import get_strategy
from utils.data_utils import fetch_historical_data


def calculate_metrics(pf: vbt.Portfolio) -> Dict[str, float]:
    """
    Bereken portfolio metrics.

    Args:
        pf: VectorBT Portfolio object

    Returns:
        Dictionary met berekende metrics
    """
    trades: vbt.Trades = pf.trades
    has_trades: bool = len(trades) > 0

    metrics: Dict[str, float] = {
        'total_return': float(pf.total_return()),
        'sharpe_ratio': float(pf.sharpe_ratio()),
        'sortino_ratio': float(pf.sortino_ratio()),
        'calmar_ratio': float(pf.calmar_ratio()),
        'max_drawdown': float(pf.max_drawdown()),
        'win_rate': float(trades.win_rate()) if has_trades else 0.0,
        'trades_count': len(trades),
        'profit_factor': float('inf'),  # Default waarde
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'max_win': 0.0,
        'max_loss': 0.0,
        'avg_duration': 0.0,
    }

    if has_trades:
        trades_df: pd.DataFrame = trades.records
        pnl_col: str = 'pnl' if 'pnl' in trades_df.columns else \
            trades_df.select_dtypes(include=[np.number]).columns[0]

        metrics['max_win'] = float(trades_df[pnl_col].max())
        metrics['max_loss'] = float(trades_df[pnl_col].min())
        metrics['avg_duration'] = float(
            trades_df['duration'].mean()) if 'duration' in trades_df.columns else 0.0

        winning_trades: pd.DataFrame = trades_df[trades_df[pnl_col] > 0]
        losing_trades: pd.DataFrame = trades_df[trades_df[pnl_col] < 0]

        if len(winning_trades) > 0:
            metrics['avg_win'] = float(winning_trades[pnl_col].mean())

        if len(losing_trades) > 0:
            metrics['avg_loss'] = float(losing_trades[pnl_col].mean())
            metrics['profit_factor'] = float(
                trades_df[pnl_col].sum() / abs(losing_trades[pnl_col].sum()))

    return metrics

def calculate_income_metrics(pf: vbt.Portfolio, metrics: Dict[str, float],
                            initial_capital: float) -> Dict[str, Any]:
    """
    Bereken inkomstenmetrics op basis van portfolio en initieel kapitaal.

    Args:
        pf: VectorBT Portfolio object
        metrics: Dictionary met bestaande metrics
        initial_capital: Initieel kapitaal

    Returns:
        Dictionary met inkomstenmetrics
    """
    monthly_returns: pd.Series = pf.returns().resample('ME').sum()
    num_years: float = len(monthly_returns) / 12

    avg_monthly_return: float = metrics['total_return'] / max(num_years, 0.1) if num_years > 0 else 0.0
    monthly_income_10k: float = 10000 * avg_monthly_return

    return {
        'monthly_returns': monthly_returns.to_dict(),
        'avg_monthly_return': avg_monthly_return,
        'monthly_income_10k': monthly_income_10k,
        'annual_income_10k': monthly_income_10k * 12,
    }

def _calculate_stop(portfolio_kwargs: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    """
    Hulpfunctie voor het berekenen en toevoegen van stop parameters aan portfolio_kwargs.

    Args:
        portfolio_kwargs: Dictionary met portfolio parameters (wordt aangepast)
        parameters: Strategie parameters
    """
    if parameters.get('use_trailing_stop', False) and 'trailing_stop_percent' in parameters:
        portfolio_kwargs['sl_trail_stop'] = parameters['trailing_stop_percent']
        logger.info(f"Trailing stop ingesteld op {parameters['trailing_stop_percent']:.2%}")

def create_visualizations(pf: vbt.Portfolio, strategy_name: str, symbol: str,
                          timeframe: Optional[str], output_path: Path,
                          timestamp: str) -> None:
    """
    Genereer en sla visualisaties op.

    Args:
        pf: VectorBT Portfolio object
        strategy_name: Naam van de strategie
        symbol: Ticker van het instrument
        timeframe: Timeframe string (bijv. 'D1')
        output_path: Pad waar visualisaties worden opgeslagen
        timestamp: Tijdstempel voor bestandsnamen
    """
    timeframe_str: str = f"_{timeframe}" if timeframe else ""

    # Equity curve
    plt.figure(figsize=(12, 6))
    try:
        pf.value().plot()
        plt.title(f"{strategy_name} Equity Curve - {symbol}")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / f"{strategy_name}_{symbol}{timeframe_str}_equity_{timestamp}.png")
        logger.info("Equity curve visualisatie opgeslagen")
    except Exception as e:
        logger.error(f"Fout bij maken equity curve plot: {str(e)}")
    finally:
        plt.close()

    # Monthly returns heatmap
    try:
        monthly_returns: pd.Series = pf.returns().resample('ME').sum()
        years: np.ndarray = monthly_returns.index.year.unique()

        if len(years) > 0:
            heatmap_data: np.ndarray = np.full((len(years), 12), np.nan)
            for i, year in enumerate(sorted(years)):
                for j, month in enumerate(range(1, 13)):
                    date: pd.Timestamp = pd.Timestamp(year=year, month=month, day=1)
                    if date in monthly_returns.index:
                        heatmap_data[i, j] = monthly_returns[date]

            plt.figure(figsize=(14, 8))
            cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list("rg", ["red", "white", "green"], N=256)
            abs_max: float = np.nanmax(np.abs(heatmap_data)) if not np.all(np.isnan(heatmap_data)) else 0.01
            # Adjusted scale to accommodate larger returns (e.g., -10% to 10%)
            im = plt.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-0.1, vmax=0.1)
            plt.colorbar(im, format='%.1f%%').set_label('Monthly Return (%)')
            plt.xticks(np.arange(12), [calendar.month_abbr[m] for m in range(1, 13)])
            plt.yticks(np.arange(len(years)), sorted(years))
            plt.title(f"{strategy_name} Monthly Returns - {symbol}")
            plt.xlabel("Month")
            plt.ylabel("Year")
            plt.savefig(output_path / f"{strategy_name}_{symbol}{timeframe_str}_monthly_returns_{timestamp}.png")
            logger.info("Monthly returns heatmap opgeslagen")
    except Exception as e:
        logger.error(f"Fout bij maken monthly returns heatmap: {str(e)}")
    finally:
        plt.close()

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
    output_path: Path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)
    timestamp: str = datetime.now().strftime("%Y%m%d")

    logger.info(f"Start backtest: {strategy_name} op {symbol} met {period_days} dagen data")
    logger.info(f"Timeframe: {timeframe}, Parameters: {parameters}")

    try:
        # Fetch data and validate
        df: Optional[pd.DataFrame] = fetch_historical_data(symbol, timeframe=timeframe, days=period_days,
                                                           end_date=end_date)
        if df is None or df.empty:
            logger.error(f"Geen geldige data voor {symbol}")
            return None, {}

        # Initialiseer RiskManager
        risk_manager: RiskManager = RiskManager(
            confidence_level=parameters.get('confidence_level', 0.95),
            max_risk=parameters.get('risk_per_trade', 0.01)
        )
        returns: pd.Series = df['close'].pct_change().dropna()
        pip_value: float = 10.0  # TODO: Maak dynamisch per symbool
        size: float = risk_manager.calculate_position_size(initial_capital, returns, pip_value)
        logger.info(f"VaR-gebaseerde positiegrootte: {size}")

        # Generate signals
        strategy = get_strategy(strategy_name, **parameters)
        entries, sl_stop, tp_stop = strategy.generate_signals(df)

        if not all(len(x) == len(df) for x in [entries, sl_stop, tp_stop]):
            raise ValueError("Signalen hebben inconsistente lengtes")

        # Map timeframe to freq for vectorbt (use lowercase 'h' to avoid FutureWarning)
        timeframe_to_freq = {
            'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
            'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w', 'MN1': '1M'
        }
        freq = timeframe_to_freq.get(str(timeframe), '1d')  # Default to '1d' if timeframe not found
        logger.info(f"Using freq: {freq} for timeframe: {timeframe}")

        # Create portfolio
        portfolio_kwargs: Dict[str, Any] = {
            'close': df['close'], 'entries': entries, 'sl_stop': sl_stop, 'tp_stop': tp_stop,
            'init_cash': initial_capital, 'fees': FEES, 'freq': freq, 'size': size,
        }

        _calculate_stop(portfolio_kwargs, parameters)
        pf: vbt.Portfolio = vbt.Portfolio.from_signals(**portfolio_kwargs)

        # Check drawdown
        current_value: float = pf.value().iloc[-1]
        max_value: float = pf.value().max()
        if risk_manager.monitor_drawdown(current_value, max_value):
            logger.warning("Maximale drawdown overschreden tijdens backtest!")

        # Calculate metrics
        metrics: Dict[str, float] = calculate_metrics(pf)
        income_metrics: Dict[str, Any] = calculate_income_metrics(pf, metrics, initial_capital)

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
        try:
            create_visualizations(pf, strategy_name, symbol, timeframe, output_path, timestamp)
        except Exception as e:
            logger.warning(f"Visualisaties konden niet worden gemaakt: {str(e)}")

        # Save trades and results
        timeframe_str: str = f"_{timeframe}" if timeframe else ""
        if len(pf.trades) > 0:
            try:
                pf.trades.records_readable.to_csv(
                    output_path / f"{strategy_name}_{symbol}{timeframe_str}_trades_{timestamp}.csv")
                logger.info("Trade gegevens opgeslagen")
            except Exception as e:
                logger.warning(f"Kon trades niet opslaan: {str(e)}")

        try:
            with open(output_path / f"{strategy_name}_{symbol}{timeframe_str}_results_{timestamp}.json", 'w') as f:
                json_metrics = {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
                                for k, v in all_metrics.items()}
                json.dump({'strategy': strategy_name, 'symbol': symbol, 'timeframe': str(timeframe),
                           'metrics': json_metrics}, f, indent=2)
                logger.info("Resultaten opgeslagen als JSON")
        except Exception as e:
            logger.warning(f"Kon JSON resultaten niet opslaan: {str(e)}")

        return pf, all_metrics

    except ValueError as ve:
        logger.error(f"Validatiefout in backtest voor {symbol}: {str(ve)}")
        return None, {}
    except Exception as e:
        logger.error(f"Backtest mislukt voor {symbol}: {str(e)}", exc_info=True)
        return None, {}

def run_monte_carlo_simulation(pf: vbt.Portfolio, n_simulations: int = 1000) -> Dict[str, Any]:
    """
    Voer Monte Carlo simulatie uit op de trades om betrouwbaarheidsintervallen te bepalen.

    Args:
        pf: VectorBT Portfolio object
        n_simulations: Aantal simulaties

    Returns:
        Dictionary met Monte Carlo resultaten
    """
    if not hasattr(pf, 'trades') or len(pf.trades) == 0:
        logger.warning("Geen trades beschikbaar voor Monte Carlo simulatie")
        return {}

    try:
        trades_df: pd.DataFrame = pf.trades.records
        pnl_col: str = 'pnl' if 'pnl' in trades_df.columns else \
            trades_df.select_dtypes(include=[np.number]).columns[0]
        trades_returns: np.ndarray = trades_df[pnl_col].values / pf.init_cash

        mc_returns: List[float] = []
        mc_drawdowns: List[float] = []

        for _ in range(n_simulations):
            sampled_returns: np.ndarray = np.random.choice(trades_returns, size=len(trades_returns), replace=True)
            cumulative: np.ndarray = np.cumprod(1 + sampled_returns)
            final_return: float = cumulative[-1] - 1
            mc_returns.append(final_return)

            peak: np.ndarray = np.maximum.accumulate(cumulative)
            drawdown: np.ndarray = (cumulative - peak) / peak
            mc_drawdowns.append(np.min(drawdown))

        return {
            'return_mean': float(np.mean(mc_returns)),
            'return_median': float(np.median(mc_returns)),
            'return_95ci_lower': float(np.percentile(mc_returns, 2.5)),
            'return_95ci_upper': float(np.percentile(mc_returns, 97.5)),
            'drawdown_mean': float(np.mean(mc_drawdowns)),
            'drawdown_worst': float(np.min(mc_drawdowns)),
            'drawdown_95ci': float(np.percentile(mc_drawdowns, 5)),
            'profit_probability': float(sum(r > 0 for r in mc_returns) / n_simulations),
        }
    except Exception as e:
        logger.error(f"Monte Carlo simulatie mislukt: {str(e)}")
        return {}

def run_walk_forward_test(strategy_name: str, parameters: Dict[str, Any], symbol: str,
                          timeframe: Optional[Union[str, int]] = None,
                          total_days: int = 1095, windows: int = 3,
                          test_percent: float = 0.3) -> Dict[str, Any]:
    """
    Voer een walk-forward test uit door historische data op te delen in train/test periodes.

    Args:
        strategy_name: Naam van de strategie
        parameters: Parameters voor de strategie
        symbol: Handelssymbool
        timeframe: Timeframe voor data
        total_days: Totaal aantal dagen data
        windows: Aantal train/test segmenten
        test_percent: Fractie van window te gebruiken voor test (0.3 = 30%)

    Returns:
        Dict met walk-forward resultaten
    """
    logger.info(f"Start walk-forward test voor {strategy_name} op {symbol}")

    now: datetime = datetime.now()
    window_days: int = total_days // windows
    test_days: int = int(window_days * test_percent)
    train_days: int = window_days - test_days

    train_results: List[Dict[str, Any]] = []
    test_results: List[Dict[str, Any]] = []

    for i in range(windows):
        window_end: datetime = now - timedelta(days=i * window_days)
        test_start: datetime = window_end - timedelta(days=test_days)
        train_start: datetime = test_start - timedelta(days=train_days)

        logger.info(f"Window {i + 1}: Train {train_start.date()} tot {test_start.date()}")
        logger.info(f"Window {i + 1}: Test {test_start.date()} tot {window_end.date()}")

        train_pf, train_metrics = run_extended_backtest(strategy_name=strategy_name,
                                                        parameters=parameters, symbol=symbol, timeframe=timeframe,
                                                        period_days=train_days, end_date=test_start)
        test_pf, test_metrics = run_extended_backtest(strategy_name=strategy_name,
                                                      parameters=parameters, symbol=symbol, timeframe=timeframe,
                                                      period_days=test_days, end_date=window_end)

        if train_metrics and test_metrics:
            train_results.append(train_metrics)
            test_results.append(test_metrics)

    if not train_results or not test_results:
        logger.warning("Onvoldoende resultaten voor walk-forward analyse")
        return {}

    train_return: float = sum(t['total_return'] for t in train_results) / len(train_results)
    test_return: float = sum(t['total_return'] for t in test_results) / len(test_results)
    train_sharpe: float = sum(t['sharpe_ratio'] for t in train_results) / len(train_results)
    test_sharpe: float = sum(t['sharpe_ratio'] for t in test_results) / len(test_results)

    robustness: Dict[str, Any] = {
        'return_ratio': test_return / max(0.001, train_return),
        'sharpe_ratio': test_sharpe / max(0.001, train_sharpe),
        'is_robust': False
    }
    robustness['is_robust'] = robustness['return_ratio'] >= 0.7 and robustness['sharpe_ratio'] >= 0.7

    results: Dict[str, Any] = {
        'windows_tested': len(test_results), 'avg_train_return': train_return,
        'avg_test_return': test_return, 'avg_train_sharpe': train_sharpe,
        'avg_test_sharpe': test_sharpe, 'robustness': robustness
    }

    logger.info(f"Walk-forward test voltooid: Strategie is {'robuust' if robustness['is_robust'] else 'NIET robuust'}")
    return results

if __name__ == "__main__":
    params: Dict[str, Any] = {'window': 60, 'std_dev': 2.5, 'risk_per_trade': 0.01, 'confidence_level': 0.95}
    pf, metrics = run_extended_backtest("BollongStrategy", params, "GER40.cash", "D1", 730, 10000)