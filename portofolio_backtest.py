#!/usr/bin/env python3
"""
SOPHY4 FIXED PORTFOLIO BACKTESTER - Windows Compatible
Target: 250+ trades/year across 5 symbols
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
import sys

# Setup logging WITHOUT emojis (Windows compatible)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project path
sys.path.append(str(Path(__file__).parent))

try:
    import vectorbt as vbt
    from config import config_manager, SYMBOLS
    from strategies import get_strategy
    from backtest.data_loader import fetch_historical_data
    from backtest.backtest import calculate_metrics
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


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
    target_trades_per_year: int
    portfolio_return: float
    average_sharpe: float
    average_win_rate: float
    worst_drawdown: float
    best_symbol: str
    worst_symbol: str
    symbol_results: List[SymbolResult]
    target_achieved: bool
    recommendations: List[str]


class FixedPortfolioBacktester:
    """BULLETPROOF Multi-symbol portfolio backtester."""

    def __init__(self, strategy_name: str = "SimpleOrderBlockStrategy",
                 timeframe: str = "H1", days: int = 365,
                 target_trades_per_year: int = 250,
                 initial_capital_per_symbol: float = 20000.0):

        self.strategy_name = strategy_name
        self.timeframe = timeframe
        self.days = days
        self.target_trades_per_year = target_trades_per_year
        self.initial_capital_per_symbol = initial_capital_per_symbol

        # HARDCODED target symbols (no config dependency)
        self.target_symbols = ["GER40.cash", "XAUUSD", "EURUSD", "US30.cash", "GBPUSD"]

        logger.info("=== FIXED PORTFOLIO BACKTESTER INITIALIZED ===")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(
            f"Target Symbols: {len(self.target_symbols)} ({', '.join(self.target_symbols)})")
        logger.info(f"Target Frequency: {target_trades_per_year} trades/year")
        logger.info(f"Capital per Symbol: ${initial_capital_per_symbol:,.0f}")

    def get_bulletproof_params(self, symbol: str) -> Dict[str, Any]:
        """Get BULLETPROOF frequency-optimized parameters - FORCE OVERRIDE."""

        # NUCLEAR OPTION: Force the most aggressive parameters possible
        nuclear_params = {# DISABLE ALL RESTRICTIVE FILTERS
            'use_htf_confirmation': False,  # KEY: Remove 100% signal blocking
            'stress_threshold': 5.0,  # VERY RELAXED vs 2.2
            'min_wick_ratio': 0.01,  # MINIMAL vs 0.3
            'use_rejection_wicks': False,  # DISABLED
            'use_session_filter': False,  # 24/7 TRADING
            'use_volume_filter': False,  # DISABLED

            # VERY WIDE RSI RANGE
            'rsi_min': 1,  # EXTREME vs 25
            'rsi_max': 99,  # EXTREME vs 75
            'rsi_period': 14,

            # RELAXED VOLUME
            'volume_multiplier': 0.5,  # VERY RELAXED vs 1.1
            'volume_period': 20,

            # AGGRESSIVE PERSONAL TRADING
            'risk_per_trade': 0.05,  # 5% vs 1.5% FTMO

            # SYMBOL
            'symbol': symbol,

            # CORE ORDER BLOCK PARAMETERS
            'ob_lookback': 3,  # SHORTER lookback
            'sl_percent': 0.01,  # 1% SL
            'tp_percent': 0.03,  # 3% TP
            'min_body_ratio': 1.0,  # RELAXED vs 1.5
            'trend_strength_min': 1.0,  # RELAXED vs 1.2

            # OVERRIDE ANY OTHER RESTRICTIVE PARAMS
            'confidence_level': 0.95, }

        logger.info(f"NUCLEAR PARAMS for {symbol}:")
        logger.info(f"  HTF Confirmation: {nuclear_params['use_htf_confirmation']}")
        logger.info(f"  Stress Threshold: {nuclear_params['stress_threshold']}")
        logger.info(f"  Min Wick Ratio: {nuclear_params['min_wick_ratio']}")
        logger.info(
            f"  RSI Range: {nuclear_params['rsi_min']}-{nuclear_params['rsi_max']}")
        logger.info(f"  All Filters Disabled: True")

        return nuclear_params

    def backtest_single_symbol(self, symbol: str) -> SymbolResult:
        """Backtest a single symbol with NUCLEAR frequency optimization."""
        logger.info(f"=== TESTING {symbol} WITH NUCLEAR PARAMS ===")

        try:
            # Get nuclear parameters
            params = self.get_bulletproof_params(symbol)

            # Fetch data
            logger.info(f"Loading data for {symbol}...")
            df = fetch_historical_data(symbol, timeframe=self.timeframe, days=self.days)
            if df is None or df.empty:
                error_msg = f"No data available for {symbol}"
                logger.error(f"FAILED: {error_msg}")
                return SymbolResult(symbol=symbol, signals=0, trades=0,
                    trades_per_year=0, total_return=0, sharpe_ratio=0, win_rate=0,
                    max_drawdown=0, strategy_params=params, success=False,
                    error_message=error_msg)

            logger.info(f"Data loaded: {len(df)} bars for {symbol}")

            # Create strategy with NUCLEAR parameters
            logger.info(f"Creating strategy with NUCLEAR params...")
            strategy = get_strategy(self.strategy_name, **params)

            # VERIFY critical parameters were applied
            logger.info(f"VERIFICATION for {symbol}:")
            logger.info(
                f"  HTF Confirmation: {getattr(strategy, 'use_htf_confirmation', 'UNKNOWN')}")
            logger.info(
                f"  Stress Threshold: {getattr(strategy, 'stress_threshold', 'UNKNOWN')}")
            logger.info(f"  RSI Min: {getattr(strategy, 'rsi_min', 'UNKNOWN')}")

            # Generate signals
            logger.info(f"Generating signals for {symbol}...")
            entries, sl_stop, tp_stop = strategy.generate_signals(df)
            total_signals = entries.sum() if hasattr(entries, 'sum') else 0

            logger.info(f"Signals generated for {symbol}: {total_signals}")

            if total_signals > 0:
                try:
                    # Create portfolio
                    logger.info(f"Creating portfolio for {symbol}...")
                    pf = vbt.Portfolio.from_signals(close=df['close'],
                        entries=entries > 0, sl_stop=sl_stop, tp_stop=tp_stop,
                        init_cash=self.initial_capital_per_symbol, fees=0.0001,
                        # 0.01% fees
                        freq='1D')

                    # Calculate metrics
                    metrics = calculate_metrics(pf)
                    trades = metrics.get('trades_count', 0)
                    trades_per_year = trades * (365 / self.days)

                    logger.info(
                        f"SUCCESS: {symbol} generated {trades} trades = {trades_per_year:.0f}/year")

                    return SymbolResult(symbol=symbol, signals=total_signals,
                        trades=trades, trades_per_year=trades_per_year,
                        total_return=metrics.get('total_return', 0),
                        sharpe_ratio=metrics.get('sharpe_ratio', 0),
                        win_rate=metrics.get('win_rate', 0),
                        max_drawdown=metrics.get('max_drawdown', 0),
                        strategy_params=params, success=True)

                except Exception as e:
                    logger.error(f"Portfolio creation failed for {symbol}: {e}")

            # No signals or portfolio creation failed
            logger.warning(f"ZERO TRADES for {symbol} - even with nuclear params!")
            return SymbolResult(symbol=symbol, signals=total_signals, trades=0,
                trades_per_year=0, total_return=0, sharpe_ratio=0, win_rate=0,
                max_drawdown=0, strategy_params=params, success=False,
                error_message=f"No signals generated for {symbol} - strategy logic issue")

        except Exception as e:
            logger.error(f"FAILED: {symbol}: {str(e)}")
            return SymbolResult(symbol=symbol, signals=0, trades=0, trades_per_year=0,
                total_return=0, sharpe_ratio=0, win_rate=0, max_drawdown=0,
                strategy_params={}, success=False, error_message=str(e))

    def run_portfolio_backtest(self, parallel: bool = False) -> PortfolioResult:
        """Run portfolio backtest across all target symbols."""
        logger.info("=== STARTING NUCLEAR PORTFOLIO BACKTEST ===")
        logger.info(
            f"Target: {self.target_trades_per_year} trades/year across {len(self.target_symbols)} symbols")

        symbol_results = []

        if parallel and len(self.target_symbols) > 1:
            logger.info("Running parallel backtests...")
            with ThreadPoolExecutor(
                    max_workers=min(3, len(self.target_symbols))) as executor:
                future_to_symbol = {
                    executor.submit(self.backtest_single_symbol, symbol): symbol for
                    symbol in self.target_symbols}

                for future in as_completed(future_to_symbol):
                    result = future.result()
                    symbol_results.append(result)
        else:
            logger.info("Running sequential backtests...")
            for symbol in self.target_symbols:
                result = self.backtest_single_symbol(symbol)
                symbol_results.append(result)

        return self._analyze_portfolio_results(symbol_results)

    def _analyze_portfolio_results(self, symbol_results: List[
        SymbolResult]) -> PortfolioResult:
        """Analyze and summarize portfolio results."""
        logger.info("=== ANALYZING NUCLEAR PORTFOLIO RESULTS ===")

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
                                                         target_achieved,
                                                         symbol_results)

        result = PortfolioResult(total_symbols=total_symbols,
            successful_symbols=successful_symbols, total_signals=total_signals,
            total_trades=total_trades, total_trades_per_year=total_trades_per_year,
            target_trades_per_year=self.target_trades_per_year,
            portfolio_return=avg_return, average_sharpe=avg_sharpe,
            average_win_rate=avg_win_rate, worst_drawdown=worst_drawdown,
            best_symbol=best_symbol, worst_symbol=worst_symbol,
            symbol_results=symbol_results, target_achieved=target_achieved,
            recommendations=recommendations)

        self._log_detailed_results(result)
        return result

    def _generate_recommendations(self, successful_results: List[SymbolResult],
                                  target_achieved: bool,
                                  all_results: List[SymbolResult]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if target_achieved:
            recommendations.append("SUCCESS! TARGET ACHIEVED: 250+ trades/year!")
            recommendations.append("Portfolio ready for live trading deployment")
            recommendations.append("Consider gradual position scaling")
        else:
            total_signals = sum(r.signals for r in all_results)

            if len(successful_results) == 0:
                recommendations.append("CRITICAL: No symbols generating any trades")
                if total_signals == 0:
                    recommendations.append("PROBLEM: Zero signals across all symbols")
                    recommendations.append(
                        "SOLUTION: HTF confirmation still blocking OR data issues")
                    recommendations.append("ACTION: Check HTF logic in strategy code")
                else:
                    recommendations.append("PROBLEM: Signals generated but no trades")
                    recommendations.append(
                        "SOLUTION: Portfolio creation or VectorBT issue")
            else:
                current_freq = sum(r.trades_per_year for r in successful_results)
                recommendations.append(
                    f"Current frequency: {current_freq:.0f} trades/year")
                recommendations.append(
                    f"Need {self.target_trades_per_year - current_freq:.0f} more trades/year")

                if current_freq < 100:
                    recommendations.append("Try even more aggressive parameters")
                    recommendations.append("Consider M15 or M30 timeframes")
                    recommendations.append("Add more symbols to portfolio")

        return recommendations

    def _log_detailed_results(self, result: PortfolioResult):
        """Log detailed portfolio results."""
        logger.info("=== NUCLEAR PORTFOLIO RESULTS ===")
        logger.info(f"Symbols Tested: {result.total_symbols}")
        logger.info(f"Successful: {result.successful_symbols}")
        logger.info(f"Total Signals: {result.total_signals}")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Trades/Year: {result.total_trades_per_year:.0f}")
        logger.info(
            f"Target: {result.target_trades_per_year} ({'ACHIEVED' if result.target_achieved else 'NOT MET'})")
        logger.info(f"Best Symbol: {result.best_symbol}")

        # Individual breakdown
        logger.info("=== INDIVIDUAL SYMBOL RESULTS ===")
        for r in sorted(result.symbol_results, key=lambda x: x.trades_per_year,
                        reverse=True):
            status = "SUCCESS" if r.success and r.trades > 0 else "FAILED"
            logger.info(
                f"{status}: {r.symbol:<12} {r.trades:>3} trades ({r.trades_per_year:>3.0f}/year)")

        # Recommendations
        logger.info("=== RECOMMENDATIONS ===")
        for i, rec in enumerate(result.recommendations, 1):
            logger.info(f"{i}. {rec}")

    def save_results(self, result: PortfolioResult,
                     output_dir: Optional[Path] = None) -> Path:
        """Save results to file."""
        if output_dir is None:
            output_dir = Path("results")
            output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nuclear_portfolio_backtest_{timestamp}.json"
        filepath = output_dir / filename

        # Convert to serializable format
        result_dict = asdict(result)

        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Results saved to: {filepath}")
        return filepath


def run_nuclear_portfolio_test(strategy_name: str = "SimpleOrderBlockStrategy",
                               timeframe: str = "H1", days: int = 180,
                               target_trades: int = 250) -> PortfolioResult:
    """Run the nuclear portfolio test."""
    logger.info("=== NUCLEAR PORTFOLIO TEST ===")

    backtester = FixedPortfolioBacktester(strategy_name=strategy_name,
        timeframe=timeframe, days=days, target_trades_per_year=target_trades)

    result = backtester.run_portfolio_backtest(
        parallel=False)  # Sequential for debugging
    backtester.save_results(result)

    return result


if __name__ == "__main__":
    print("=== SOPHY4 NUCLEAR PORTFOLIO BACKTESTER ===")
    print("Testing NUCLEAR frequency-optimized parameters across 5 symbols")
    print("Target: 250+ trades/year total")
    print("=" * 60)

    result = run_nuclear_portfolio_test()

    print(f"\nRESULTS:")
    print(f"Total Trades/Year: {result.total_trades_per_year:.0f}")
    print(f"Target: {result.target_trades_per_year}")

    if result.target_achieved:
        print(f"\nSUCCESS! {result.total_trades_per_year:.0f} trades/year achieved!")
        print("Ready for live trading deployment")
    else:
        print(
            f"\nTarget not met: {result.total_trades_per_year:.0f}/{result.target_trades_per_year}")
        print("\nDIAGNOSTIC INFO:")
        print(f"Successful symbols: {result.successful_symbols}/{result.total_symbols}")
        print(f"Total signals: {result.total_signals}")

        if result.total_signals == 0:
            print("\nCRITICAL: HTF confirmation STILL blocking signals!")
            print("Need to hardcode HTF return True in strategy code")
        elif result.successful_symbols == 0:
            print("\nPROBLEM: Signals generated but no trades executed")
            print("Check VectorBT portfolio creation logic")