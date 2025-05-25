#!/usr/bin/env python
"""
Quick test script to identify the exact problem
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from backtest.backtest import run_extended_backtest
from config import logger


def quick_debug_test():
    """Test the strategy with minimal filters to find the problem"""

    logger.info("🔍 STARTING QUICK DEBUG TEST")
    logger.info("=" * 60)

    # Test 1: Debug version with ALL filters disabled
    logger.info("TEST 1: Debug version with filters disabled")

    params = {'symbol': 'GER40.cash', 'use_rejection_wicks': False,
        'use_session_filter': False, 'use_htf_confirmation': False,
        # DISABLED - likely culprit
        'stress_threshold': 5.0,  # Very relaxed
        'volume_multiplier': 1.0,  # Very relaxed
        'rsi_min': 20, 'rsi_max': 80}

    try:
        pf, metrics = run_extended_backtest(strategy_name="SimpleOrderBlockStrategy",
            parameters=params, symbol="GER40.cash", timeframe="H1", period_days=365,
            # Shorter period for faster test
            silent=False)

        if metrics:
            logger.info(f"✅ Test 1 Results:")
            logger.info(f"   Trades: {metrics.get('trades_count', 0)}")
            logger.info(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
            logger.info(f"   Total Return: {metrics.get('total_return', 0):.2%}")

    except Exception as e:
        logger.error(f"Test 1 failed: {e}")

    logger.info("=" * 60)

    # Test 2: Check if HTF confirmation is the culprit
    logger.info("TEST 2: Enable HTF confirmation to see if it blocks signals")

    params['use_htf_confirmation'] = True  # Enable the likely culprit

    try:
        pf, metrics = run_extended_backtest(strategy_name="SimpleOrderBlockStrategy",
            parameters=params, symbol="GER40.cash", timeframe="H1", period_days=365,
            silent=False)

        if metrics:
            logger.info(f"✅ Test 2 Results (HTF enabled):")
            logger.info(f"   Trades: {metrics.get('trades_count', 0)}")
            logger.info(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
            logger.info(f"   Total Return: {metrics.get('total_return', 0):.2%}")

            if metrics.get('trades_count', 0) == 0:
                logger.error("🚨 HTF CONFIRMATION IS BLOCKING ALL SIGNALS!")
                logger.error(
                    "   Solution: Disable HTF confirmation or use different timeframe")

    except Exception as e:
        logger.error(f"Test 2 failed: {e}")

    logger.info("=" * 60)
    logger.info("🎯 RECOMMENDATIONS:")
    logger.info("1. If Test 1 has trades but Test 2 has 0 trades:")
    logger.info("   → HTF confirmation is the problem (GER40 is bearish)")
    logger.info("2. If both tests have 0 trades:")
    logger.info("   → Base filters are too strict")
    logger.info("3. Quick fix: Set use_htf_confirmation=False in params")


if __name__ == "__main__":
    quick_debug_test()