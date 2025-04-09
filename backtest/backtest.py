import calendar
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from matplotlib.colors import LinearSegmentedColormap

from config import INITIAL_CAPITAL, FEES, OUTPUT_DIR, logger
from ftmo_compliance.ftmo_check import check_ftmo_compliance
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
    trades = pf.trades
    has_trades = len(trades) > 0

    # Log trades-structuur voor debugging
    logger.debug(
        f"Trades structuur: {trades.records.columns.tolist() if has_trades else 'Geen trades'}")

    metrics = {'total_return': float(pf.total_return()),
        'sharpe_ratio': float(pf.sharpe_ratio()),
        'sortino_ratio': float(pf.sortino_ratio()),
        'calmar_ratio': float(pf.calmar_ratio()),
        'max_drawdown': float(pf.max_drawdown()),
        'win_rate': float(trades.win_rate()) if has_trades else 0.0,
        'trades_count': len(trades), 'profit_factor': float('inf'),  # Default waarde
        'avg_win': 0.0,  # Default waarde
        'avg_loss': 0.0,  # Default waarde
        'max_win': 0.0,  # Default waarde
        'max_loss': 0.0,  # Default waarde
        'avg_duration': 0.0,  # Default waarde
    }

    if has_trades:
        trades_df = trades.records
        pnl_col = 'pnl' if 'pnl' in trades_df.columns else \
        trades_df.select_dtypes(include=[np.number]).columns[0]
        logger.debug(f"Gebruikte PNL-kolom: {pnl_col}")

        metrics['max_win'] = float(trades_df[pnl_col].max())
        metrics['max_loss'] = float(trades_df[pnl_col].min())
        metrics['avg_duration'] = float(
            trades_df['duration'].mean()) if 'duration' in trades_df.columns else 0.0

        winning_trades = trades_df[trades_df[pnl_col] > 0]
        losing_trades = trades_df[trades_df[pnl_col] < 0]

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
    Bereken inkomstenmetrics op basis van portfolio en initial kapitaal.

    Args:
        pf: VectorBT Portfolio object
        metrics: Dictionary met bestaande metrics
        initial_capital: Initieel kapitaal

    Returns:
        Dictionary met inkomstenmetrics
    """
    monthly_returns = pf.returns().resample('ME').sum()  # 'M' vervangen door 'ME'
    num_years = len(monthly_returns) / 12

    avg_monthly_return = metrics['total_return'] / max(num_years,
                                                       0.1) if num_years > 0 else 0.0
    monthly_income_10k = 10000 * avg_monthly_return

    return {'monthly_returns': monthly_returns.to_dict(),
        'avg_monthly_return': avg_monthly_return,
        'monthly_income_10k': monthly_income_10k,
        'annual_income_10k': monthly_income_10k * 12, }


def _calculate_stop(portfolio_kwargs: Dict[str, Any],
                    parameters: Dict[str, Any]) -> None:
    """
    Hulpfunctie voor het berekenen en toevoegen van stop parameters aan portfolio_kwargs.
    Centraliseert code die anders herhaald zou worden.

    Args:
        portfolio_kwargs: Dictionary met portfolio parameters (wordt aangepast)
        parameters: Strategie parameters
    """
    if parameters.get('use_trailing_stop',
                      False) and 'trailing_stop_percent' in parameters:
        portfolio_kwargs['sl_trail_stop'] = parameters['trailing_stop_percent']
        logger.info(
            f"Trailing stop ingesteld op {parameters['trailing_stop_percent']:.2%}")


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
    timeframe_str = f"_{timeframe}" if timeframe else ""

    # Equity curve - Fixed for newer VectorBT versions
    plt.figure(figsize=(12, 6))
    try:
        # Direct plotting of portfolio value (compatible approach)
        pf.value().plot()
        plt.title(f"{strategy_name} Equity Curve - {symbol}")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            output_path / f"{strategy_name}_{symbol}{timeframe_str}_equity_{timestamp}.png")
        logger.info(f"Equity curve visualisatie opgeslagen")
    except Exception as e:
        logger.error(f"Fout bij maken equity curve plot: {str(e)}")
    finally:
        plt.close()

    # Monthly returns heatmap
    try:
        monthly_returns = pf.returns().resample('ME').sum()
        years = monthly_returns.index.year.unique()

        if len(years) > 0:
            heatmap_data = np.full((len(years), 12), np.nan)
            for i, year in enumerate(sorted(years)):
                for j, month in enumerate(range(1, 13)):
                    date = pd.Timestamp(year=year, month=month, day=1)
                    if date in monthly_returns.index:
                        heatmap_data[i, j] = monthly_returns[date]

            plt.figure(figsize=(14, 8))
            cmap = LinearSegmentedColormap.from_list("rg", ["red", "white", "green"], N=256)
            abs_max = np.nanmax(np.abs(heatmap_data)) if not np.all(
                np.isnan(heatmap_data)) else 0.01
            im = plt.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-abs_max,
                            vmax=abs_max)
            plt.colorbar(im, format='%.1f%%').set_label('Monthly Return (%)')
            plt.xticks(np.arange(12), [calendar.month_abbr[m] for m in range(1, 13)])
            plt.yticks(np.arange(len(years)), sorted(years))
            plt.title(f"{strategy_name} Monthly Returns - {symbol}")
            plt.xlabel("Month")
            plt.ylabel("Year")
            plt.savefig(
                output_path / f"{strategy_name}_{symbol}{timeframe_str}_monthly_returns_{timestamp}.png")
            logger.info(f"Monthly returns heatmap opgeslagen")
    except Exception as e:
        logger.error(f"Fout bij maken monthly returns heatmap: {str(e)}")
    finally:
        plt.close()


def run_extended_backtest(strategy_name: str, parameters: Dict[str, Any], symbol: str,
                          timeframe: Optional[Union[str, int]] = None,
                          period_days: int = 1095,
                          initial_capital: float = INITIAL_CAPITAL) -> Tuple[
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

    Returns:
        Tuple van (Portfolio-object of None, dictionary met metrics).
    """
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d")

    logger.info(
        f"Start backtest: {strategy_name} op {symbol} met {period_days} dagen data")
    logger.info(f"Timeframe: {timeframe}, Parameters: {parameters}")

    try:
        # Fetch data and validate
        df = fetch_historical_data(symbol, timeframe=timeframe, days=period_days)
        if df is None or df.empty:
            logger.error(f"Geen geldige data voor {symbol}")
            return None, {}

        # Generate signals
        strategy = get_strategy(strategy_name, **parameters)
        entries, sl_stop, tp_stop = strategy.generate_signals(df)

        if not all(len(x) == len(df) for x in [entries, sl_stop, tp_stop]):
            raise ValueError("Signalen hebben inconsistente lengtes")

        # Position sizing
        size = None
        if 'risk_per_trade' in parameters:
            size = parameters['risk_per_trade'] / sl_stop.replace(0, np.inf).clip(
                lower=0.01)
            logger.info(
                f"Positiegrootte berekend op basis van risk_per_trade={parameters['risk_per_trade']}")

        # Create portfolio
        portfolio_kwargs = {'close': df['close'], 'entries': entries,
            'sl_stop': sl_stop, 'tp_stop': tp_stop, 'init_cash': initial_capital,
            'fees': FEES, 'freq': '1D', 'size': size, }

        _calculate_stop(portfolio_kwargs, parameters)
        pf = vbt.Portfolio.from_signals(**portfolio_kwargs)

        # Calculate metrics
        metrics = calculate_metrics(pf)
        income_metrics = calculate_income_metrics(pf, metrics, initial_capital)

        compliant, profit_target = check_ftmo_compliance(pf, metrics)
        all_metrics = {**metrics, **income_metrics, 'ftmo_compliant': compliant,
                       'profit_target_reached': profit_target}

        # Log results
        logger.info(
            f"\n===== BACKTEST RESULTATEN VOOR {strategy_name} OP {symbol} =====")
        logger.info(f"Totaal rendement: {metrics['total_return']:.2%}")
        logger.info(
            f"Sharpe: {metrics['sharpe_ratio']:.2f}, Max drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(
            f"Win rate: {metrics['win_rate']:.2%}, Trades: {metrics['trades_count']}")
        logger.info(
            f"FTMO compliant: {'JA' if compliant else 'NEE'}, Profit target bereikt: {'JA' if profit_target else 'NEE'}")

        # Try to create visualizations but continue even if it fails
        try:
            create_visualizations(pf, strategy_name, symbol, timeframe, output_path,
                                timestamp)
        except Exception as e:
            logger.warning(f"Visualisaties konden niet worden gemaakt: {str(e)}")
            # Continue with the rest of the process

        # Save trades and results
        timeframe_str = f"_{timeframe}" if timeframe else ""
        if len(pf.trades) > 0:
            try:
                pf.trades.records_readable.to_csv(
                    output_path / f"{strategy_name}_{symbol}{timeframe_str}_trades_{timestamp}.csv")
                logger.info("Trade gegevens opgeslagen")
            except Exception as e:
                logger.warning(f"Kon trades niet opslaan: {str(e)}")

        # Save JSON results
        try:
            with open(
                    output_path / f"{strategy_name}_{symbol}{timeframe_str}_results_{timestamp}.json",
                    'w') as f:
                json_metrics = {k: (
                    str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
                                for k, v in all_metrics.items()}
                json.dump({'strategy': strategy_name, 'symbol': symbol,
                        'timeframe': str(timeframe), 'metrics': json_metrics}, f,
                        indent=2)
                logger.info("Resultaten opgeslagen als JSON")
        except Exception as e:
            logger.warning(f"Kon JSON resultaten niet opslaan: {str(e)}")

        return pf, all_metrics

    except Exception as e:
        logger.error(f"Backtest mislukt voor {symbol}: {str(e)}", exc_info=True)
        return None, {}


if __name__ == "__main__":
    # Voor debugging: draai een simpele test
    params = {'window': 60, 'std_dev': 2.5, 'risk_per_trade': 0.01}
    pf, metrics = run_extended_backtest("BollongStrategy", params, "GER40.cash", "D1",
                                        730, 10000)