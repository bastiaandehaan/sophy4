# portfolio_backtest.py - Windows Compatible Version (No Emojis)
"""
Multi-Symbol Portfolio Backtesting Framework - Windows Compatible
Target: 250+ trades/year across 5 symbols (60+ trades per symbol)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import vectorbt as vbt

# Import our updated config
from config import (config_manager, logger, SYMBOLS)
from strategies import get_strategy
from backtest.data_loader import fetch_historical_data
from backtest.backtest import calculate_metrics, calculate_income_metrics
from ftmo_compliance.ftmo_check import check_ftmo_compliance


@dataclass
class SymbolResult:
    """Results for a single symbol."""
    symbol: str
    signals: int
    trades: int
    trades_per_year: float
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    strategy_params: Dict[str, Any]
    success: bool = True
    error_message: str = ""


@dataclass
class PortfolioResult:
    """Aggregated portfolio results."""
    total_symbols: int
    successful_symbols: int
    total_signals: int
    total_trades: int
    total_trades_per_year: float
    target_trades_per_year: int  # FIXED: Add missing attribute
    portfolio_return: float
    average_sharpe: float
    average_win_rate: float
    worst_drawdown: float
    best_symbol: str
    worst_symbol: str
    symbol_results: List[SymbolResult]
    target_achieved: bool
    recommendations: List[str]


class PortfolioBacktester:
    """Multi-symbol portfolio backtesting for maximum trade frequency."""

    def __init__(self, strategy_name: str = "SimpleOrderBlockStrategy",
                 timeframe: str = "H1", days: int = 365,
                 target_trades_per_year: int = 250,
                 initial_capital_per_symbol: float = 20000.0):

        self.strategy_name = strategy_name
        self.timeframe = timeframe
        self.days = days
        self.target_trades_per_year = target_trades_per_year
        self.initial_capital_per_symbol = initial_capital_per_symbol

        # Get target symbols
        self.target_symbols = SYMBOLS

        logger.info("=== PORTFOLIO BACKTESTER INITIALIZED ===")
        logger.info(f"   Strategy: {strategy_name}")
        logger.info(
            f"   Target Symbols: {len(self.target_symbols)} ({', '.join(self.target_symbols)})")
        logger.info(f"   Target Frequency: {target_trades_per_year} trades/year")
        logger.info(f"   Capital per Symbol: ${initial_capital_per_symbol:,.0f}")

    def get_frequency_optimized_params(self, symbol: str) -> Dict[str, Any]:
        """Get frequency-optimized parameters for the symbol - FORCE OVERRIDE."""
        # CRITICAL: Force the frequency-optimized parameters regardless of config
        frequency_params = {# FORCE DISABLE all restrictive filters
            'use_htf_confirmation': False,  # KEY: Disable HTF blocking
            'stress_threshold': 4.0,  # RELAXED: vs 2.2 FTMO
            'min_wick_ratio': 0.05,  # MINIMAL: vs 0.3 FTMO
            'use_rejection_wicks': False,  # DISABLE: no wick requirement
            'use_session_filter': False,  # DISABLE: 24/7 trading

            # RSI - VERY WIDE range
            'rsi_min': 5,  # vs 25 FTMO
            'rsi_max': 95,  # vs 75 FTMO

            # Volume - RELAXED
            'volume_multiplier': 0.8,  # vs 1.1 FTMO
            'use_volume_filter': False,  # Optional filter

            # Risk - PERSONAL TRADING
            'risk_per_trade': 0.05,  # 5% vs 1.5%

            # Symbol
            'symbol': symbol,

            # Core order block parameters
            'ob_lookback': 5, 'sl_percent': 0.01, 'tp_percent': 0.03, 'rsi_period': 14,
            'volume_period': 20, 'min_body_ratio': 1.5}

        logger.info(f"   FORCED frequency params for {symbol}:")
        logger.info(
            f"     HTF Confirmation: {frequency_params['use_htf_confirmation']}")
        logger.info(f"     Stress Threshold: {frequency_params['stress_threshold']}")
        logger.info(f"     Min Wick Ratio: {frequency_params['min_wick_ratio']}")
        logger.info(
            f"     RSI Range: {frequency_params['rsi_min']}-{frequency_params['rsi_max']}")
        logger.info(f"     Risk per Trade: {frequency_params['risk_per_trade']:.1%}")

        return frequency_params

    def backtest_single_symbol(self, symbol: str) -> SymbolResult:
        """Backtest a single symbol with frequency optimization."""
        logger.info(f"=== Testing {symbol} ===")

        try:
            # Get frequency-optimized parameters - FORCED OVERRIDE
            params = self.get_frequency_optimized_params(symbol)

            # Fetch data
            df = fetch_historical_data(symbol, timeframe=self.timeframe, days=self.days)
            if df is None or df.empty:
                return SymbolResult(symbol=symbol, signals=0, trades=0,
                    trades_per_year=0, total_return=0, sharpe_ratio=0, win_rate=0,
                    max_drawdown=0, strategy_params=params, success=False,
                    error_message=f"No data available for {symbol}")

            logger.info(f"   Data loaded: {len(df)} bars")

            # Create strategy with FORCED frequency-optimized parameters
            strategy = get_strategy(self.strategy_name, **params)

            # VERIFY the parameters were applied correctly
            logger.info(f"   VERIFICATION - Strategy parameters:")
            logger.info(
                f"     HTF Confirmation: {getattr(strategy, 'use_htf_confirmation', 'N/A')}")
            logger.info(
                f"     Stress Threshold: {getattr(strategy, 'stress_threshold', 'N/A')}")
            logger.info(
                f"     Min Wick Ratio: {getattr(strategy, 'min_wick_ratio', 'N/A')}")
            logger.info(
                f"     Risk per Trade: {getattr(strategy, 'risk_per_trade', 'N/A')}")

            # Generate signals
            entries, sl_stop, tp_stop = strategy.generate_signals(df)
            total_signals = entries.sum()

            logger.info(f"   Signals generated: {total_signals}")

            # Create portfolio if we have signals
            if total_signals > 0:
                try:
                    pf = vbt.Portfolio.from_signals(close=df['close'],
                        entries=entries > 0, sl_stop=sl_stop, tp_stop=tp_stop,
                        init_cash=self.initial_capital_per_symbol,
                        fees=config_manager.trading.fees, freq='1D'
                        # Daily frequency for annualization
                    )

                    # Calculate metrics
                    metrics = calculate_metrics(pf)
                    trades = metrics.get('trades_count', 0)
                    trades_per_year = trades * (365 / self.days)

                    logger.info(
                        f"   SUCCESS: {symbol}: {trades} trades = {trades_per_year:.0f}/year")

                    return SymbolResult(symbol=symbol, signals=total_signals,
                        trades=trades, trades_per_year=trades_per_year,
                        total_return=metrics.get('total_return', 0),
                        sharpe_ratio=metrics.get('sharpe_ratio', 0),
                        win_rate=metrics.get('win_rate', 0),
                        max_drawdown=metrics.get('max_drawdown', 0),
                        strategy_params=params, success=True)

                except Exception as e:
                    logger.error(f"   Portfolio creation failed for {symbol}: {e}")

            # No signals case
            logger.warning(f"   NO SIGNALS for {symbol} - filters too restrictive")
            return SymbolResult(symbol=symbol, signals=total_signals, trades=0,
                trades_per_year=0, total_return=0, sharpe_ratio=0, win_rate=0,
                max_drawdown=0, strategy_params=params, success=False,
                error_message=f"No signals generated for {symbol} - check filter settings")

        except Exception as e:
            logger.error(f"   FAILED: {symbol}: {str(e)}")
            return SymbolResult(symbol=symbol, signals=0, trades=0, trades_per_year=0,
                total_return=0, sharpe_ratio=0, win_rate=0, max_drawdown=0,
                strategy_params={}, success=False, error_message=str(e))

    def run_portfolio_backtest(self, parallel: bool = True) -> PortfolioResult:
        """Run portfolio backtest across all target symbols."""
        logger.info("=== STARTING PORTFOLIO BACKTEST ===")
        logger.info(
            f"Target: {self.target_trades_per_year} trades/year across {len(self.target_symbols)} symbols")

        symbol_results = []

        if parallel and len(self.target_symbols) > 1:
            # Parallel execution for speed
            logger.info("Running parallel backtests...")
            with ThreadPoolExecutor(
                    max_workers=min(4, len(self.target_symbols))) as executor:
                future_to_symbol = {
                    executor.submit(self.backtest_single_symbol, symbol): symbol for
                    symbol in self.target_symbols}

                for future in as_completed(future_to_symbol):
                    result = future.result()
                    symbol_results.append(result)
        else:
            # Sequential execution
            logger.info("Running sequential backtests...")
            for symbol in self.target_symbols:
                result = self.backtest_single_symbol(symbol)
                symbol_results.append(result)

        # Analyze portfolio results
        return self._analyze_portfolio_results(symbol_results)

    def _analyze_portfolio_results(self, symbol_results: List[
        SymbolResult]) -> PortfolioResult:
        """Analyze and summarize portfolio results."""
        logger.info("=== ANALYZING PORTFOLIO RESULTS ===")

        successful_results = [r for r in symbol_results if r.success and r.trades > 0]

        # Calculate portfolio metrics
        total_symbols = len(symbol_results)
        successful_symbols = len(successful_results)
        total_signals = sum(r.signals for r in symbol_results)
        total_trades = sum(r.trades for r in successful_results)
        total_trades_per_year = sum(r.trades_per_year for r in successful_results)

        # Average metrics
        if successful_results:
            avg_return = np.mean([r.total_return for r in successful_results])
            avg_sharpe = np.mean(
                [r.sharpe_ratio for r in successful_results if r.sharpe_ratio > 0])
            avg_win_rate = np.mean([r.win_rate for r in successful_results])
            worst_drawdown = min([r.max_drawdown for r in successful_results])

            # Best and worst symbols
            best_symbol = max(successful_results,
                              key=lambda x: x.trades_per_year).symbol
            worst_symbol = min(successful_results,
                               key=lambda x: x.trades_per_year).symbol
        else:
            avg_return = avg_sharpe = avg_win_rate = worst_drawdown = 0
            best_symbol = worst_symbol = "None"

        # Check if target achieved
        target_achieved = total_trades_per_year >= self.target_trades_per_year

        # Generate recommendations
        recommendations = self._generate_recommendations(successful_results,
                                                         target_achieved)

        result = PortfolioResult(total_symbols=total_symbols,
            successful_symbols=successful_symbols, total_signals=total_signals,
            total_trades=total_trades, total_trades_per_year=total_trades_per_year,
            target_trades_per_year=self.target_trades_per_year,  # FIXED: Set the target
            portfolio_return=avg_return, average_sharpe=avg_sharpe,
            average_win_rate=avg_win_rate, worst_drawdown=worst_drawdown,
            best_symbol=best_symbol, worst_symbol=worst_symbol,
            symbol_results=symbol_results, target_achieved=target_achieved,
            recommendations=recommendations)

        # Log detailed results
        self._log_detailed_results(result)

        return result

    def _generate_recommendations(self, successful_results: List[SymbolResult],
                                  target_achieved: bool) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        if target_achieved:
            recommendations.append(
                "SUCCESS! TARGET ACHIEVED: Portfolio ready for live trading")
            recommendations.append("Consider gradually scaling position sizes")
            recommendations.append("Monitor correlation between symbols")
        else:
            if len(successful_results) == 0:
                recommendations.append("CRITICAL: No symbols generating trades")
                recommendations.append(
                    "SOLUTION: HTF confirmation is still blocking signals")
                recommendations.append(
                    "ACTION: Manually disable HTF filter in strategy code")
                recommendations.append("CHECK: Verify config.py was updated correctly")
            elif len(successful_results) < len(SYMBOLS) // 2:
                recommendations.append("Only few symbols working - expand symbol list")
                recommendations.append(
                    "Further relax parameters for underperforming symbols")
            else:
                total_frequency = sum(r.trades_per_year for r in successful_results)
                recommendations.append(f"Current: {total_frequency:.0f} trades/year")
                recommendations.append("Optimize underperforming symbols")

                if total_frequency < self.target_trades_per_year * 0.8:
                    recommendations.append("Try even more aggressive parameters")
                    recommendations.append("Consider additional timeframes (M15, M30)")

        return recommendations

    def _log_detailed_results(self, result: PortfolioResult):
        """Log detailed portfolio results."""
        logger.info("=== PORTFOLIO RESULTS SUMMARY ===")
        logger.info(f"Symbols Tested: {result.total_symbols}")
        logger.info(f"Successful: {result.successful_symbols}")
        logger.info(f"Total Signals: {result.total_signals}")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Trades/Year: {result.total_trades_per_year:.0f}")
        logger.info(
            f"Target: {result.target_trades_per_year} ({'ACHIEVED' if result.target_achieved else 'NOT MET'})")
        logger.info(f"Avg Return: {result.portfolio_return:.2%}")
        logger.info(f"Avg Sharpe: {result.average_sharpe:.2f}")
        logger.info(f"Avg Win Rate: {result.average_win_rate:.1%}")
        logger.info(f"Best Symbol: {result.best_symbol}")
        logger.info(f"Worst Symbol: {result.worst_symbol}")

        # Individual symbol breakdown
        logger.info("=== INDIVIDUAL SYMBOL RESULTS ===")
        for r in sorted(result.symbol_results, key=lambda x: x.trades_per_year,
                        reverse=True):
            status = "SUCCESS" if r.success and r.trades > 0 else "FAILED"
            logger.info(
                f"{status}: {r.symbol:<12} {r.trades:>3} trades ({r.trades_per_year:>3.0f}/year) "
                f"Return: {r.total_return:>6.1%} Sharpe: {r.sharpe_ratio:>4.1f}")

        # Recommendations
        logger.info("=== RECOMMENDATIONS ===")
        for i, rec in enumerate(result.recommendations, 1):
            logger.info(f"{i}. {rec}")

    def save_results(self, result: PortfolioResult,
                     output_dir: Optional[Path] = None) -> Path:
        """Save portfolio results to file."""
        if output_dir is None:
            output_dir = config_manager.get_output_dir()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_backtest_{self.strategy_name}_{timestamp}.json"
        filepath = output_dir / filename

        # Convert to serializable format
        result_dict = asdict(result)

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Results saved to: {filepath}")
        return filepath


def run_quick_portfolio_test(strategy_name: str = "SimpleOrderBlockStrategy",
                             timeframe: str = "H1", days: int = 365,
                             target_trades: int = 250) -> PortfolioResult:
    """Quick portfolio test function."""
    logger.info("=== QUICK PORTFOLIO TEST ===")

    backtester = PortfolioBacktester(strategy_name=strategy_name, timeframe=timeframe,
        days=days, target_trades_per_year=target_trades)

    result = backtester.run_portfolio_backtest(parallel=True)
    backtester.save_results(result)

    return result


if __name__ == "__main__":
    # Test the portfolio framework
    print("=== SOPHY4 PORTFOLIO BACKTESTER ===")
    print("Testing frequency-optimized parameters across 5 symbols")
    print("=" * 60)

    result = run_quick_portfolio_test()

    if result.target_achieved:
        print(f"\nSUCCESS! {result.total_trades_per_year:.0f} trades/year achieved!")
        print("Ready for live trading deployment")
    else:
        print(
            f"\nTarget not met: {result.total_trades_per_year:.0f}/{result.target_trades_per_year}")
        print("Check recommendations for parameter optimization")

        # Show key diagnostic info
        print(f"\nDIAGNOSTIC INFO:")
        print(f"Symbols tested: {result.total_symbols}")
        print(
            f"Symbols with signals: {len([r for r in result.symbol_results if r.signals > 0])}")
        print(f"Symbols with trades: {result.successful_symbols}")

        if result.successful_symbols == 0:
            print("\nCRITICAL: HTF confirmation is still blocking all signals!")
            print("The configuration changes are not being applied correctly.")
            print("Check that the updated config.py is being used.")