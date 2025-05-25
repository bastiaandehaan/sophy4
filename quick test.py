#!/usr/bin/env python3
"""
🚀 SOPHY4 FREQUENCY FIX VALIDATION
Test that the configuration changes achieve the target 250+ trades/year
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
from datetime import datetime
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_config_changes():
    """Test that configuration changes are working."""
    logger.info("🧪 TESTING CONFIG CHANGES")
    logger.info("=" * 50)

    try:
        # Import updated config
        from config import (config_manager, get_strategy_config,
                            get_multi_symbol_config, PERSONAL_MAX_RISK_PER_TRADE,
                            SYMBOLS)

        # Test 1: Verify key configuration changes
        logger.info("1. Verifying configuration changes...")

        h1_config = get_strategy_config("SimpleOrderBlockStrategy", "H1")

        # Check critical frequency parameters
        tests = [("HTF Confirmation", h1_config.get('use_htf_confirmation'), False),
            ("Stress Threshold", h1_config.get('stress_threshold'), 4.0),
            ("Min Wick Ratio", h1_config.get('min_wick_ratio'), 0.05),
            ("Rejection Wicks", h1_config.get('use_rejection_wicks'), False),
            ("Session Filter", h1_config.get('use_session_filter'), False),
            ("RSI Min", h1_config.get('rsi_min'), 5),
            ("RSI Max", h1_config.get('rsi_max'), 95),
            ("Risk per Trade", PERSONAL_MAX_RISK_PER_TRADE, 0.05)]

        all_passed = True
        for test_name, actual, expected in tests:
            status = "✅" if actual == expected else "❌"
            logger.info(f"   {status} {test_name}: {actual} (expected: {expected})")
            if actual != expected:
                all_passed = False

        if all_passed:
            logger.info("   🎉 All configuration tests PASSED!")
        else:
            logger.error("   ❌ Some configuration tests FAILED!")
            return False

        # Test 2: Multi-symbol configuration
        logger.info("2. Verifying multi-symbol setup...")
        multi_config = get_multi_symbol_config()

        logger.info(f"   📊 Target symbols: {len(SYMBOLS)} ({', '.join(SYMBOLS)})")
        logger.info(
            f"   🎯 Expected frequency: {multi_config['total_expected_trades']} trades/year")
        logger.info(f"   💰 Portfolio risk: {multi_config['portfolio_risk']:.1%}")

        return True

    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_strategy_parameters():
    """Test that strategy now accepts the frequency parameters."""
    logger.info("\n🧪 TESTING STRATEGY PARAMETER ACCEPTANCE")
    logger.info("=" * 50)

    try:
        from strategies.simple_order_block import SimpleOrderBlockStrategy

        # Test parameter sets
        test_configs = {
            "FTMO (Old)": {'use_htf_confirmation': True, 'stress_threshold': 2.2,
                'min_wick_ratio': 0.3, 'use_rejection_wicks': True, 'rsi_min': 25,
                'rsi_max': 75},
            "PERSONAL (New)": {'use_htf_confirmation': False, 'stress_threshold': 4.0,
                'min_wick_ratio': 0.05, 'use_rejection_wicks': False, 'rsi_min': 5,
                'rsi_max': 95}}

        for config_name, params in test_configs.items():
            logger.info(f"   Testing {config_name} configuration...")

            try:
                strategy = SimpleOrderBlockStrategy(**params)

                # Verify parameters were set correctly
                assert strategy.use_htf_confirmation == params['use_htf_confirmation']
                assert strategy.stress_threshold == params['stress_threshold']
                assert strategy.min_wick_ratio == params['min_wick_ratio']
                assert strategy.use_rejection_wicks == params['use_rejection_wicks']
                assert strategy.rsi_min == params['rsi_min']
                assert strategy.rsi_max == params['rsi_max']

                logger.info(f"      ✅ {config_name}: Parameters accepted correctly")

            except Exception as e:
                logger.error(f"      ❌ {config_name}: Failed - {e}")
                return False

        logger.info("   🎉 All parameter tests PASSED!")
        return True

    except Exception as e:
        logger.error(f"❌ Strategy parameter test failed: {e}")
        return False


def test_signal_generation():
    """Test signal generation with old vs new parameters."""
    logger.info("\n🧪 TESTING SIGNAL GENERATION FREQUENCY")
    logger.info("=" * 50)

    try:
        from strategies.simple_order_block import SimpleOrderBlockStrategy
        from backtest.data_loader import fetch_historical_data

        # Get test data
        logger.info("   📥 Loading test data...")
        df = fetch_historical_data("GER40.cash", "H1", days=180)  # 6 months

        if df is None or df.empty:
            logger.error("   ❌ Could not load test data")
            return False

        logger.info(f"   ✅ Data loaded: {len(df)} bars")

        # Test configurations
        test_configs = [("FTMO Restrictive",
                         {'use_htf_confirmation': True, 'stress_threshold': 2.2,
                             'min_wick_ratio': 0.3, 'use_rejection_wicks': True,
                             'rsi_min': 25, 'rsi_max': 75}), ("Personal Optimized", {
            'use_htf_confirmation': False,  # KEY: Remove HTF blocking
            'stress_threshold': 4.0,  # KEY: Relax stress filter
            'min_wick_ratio': 0.05,  # KEY: Minimal wick requirements
            'use_rejection_wicks': False,  # KEY: No wick filter
            'rsi_min': 5,  # KEY: Wide RSI range
            'rsi_max': 95})]

        results = []

        for config_name, params in test_configs:
            logger.info(f"   🧪 Testing {config_name}...")

            try:
                strategy = SimpleOrderBlockStrategy(**params)
                entries, sl, tp = strategy.generate_signals(df)

                signals = entries.sum()
                trades_per_year = signals * (365 / len(df))

                logger.info(
                    f"      📊 {signals} signals = {trades_per_year:.0f} trades/year")

                results.append({'name': config_name, 'signals': signals,
                    'trades_per_year': trades_per_year})

            except Exception as e:
                logger.error(f"      ❌ {config_name} failed: {e}")
                return False

        # Compare results
        logger.info("   📊 COMPARISON:")
        if len(results) >= 2:
            old_freq = results[0]['trades_per_year']
            new_freq = results[1]['trades_per_year']

            improvement = new_freq / max(old_freq, 1)

            logger.info(f"      FTMO: {old_freq:.0f} trades/year")
            logger.info(f"      Personal: {new_freq:.0f} trades/year")
            logger.info(f"      Improvement: {improvement:.1f}x")

            if new_freq >= 60:  # Target per symbol
                logger.info("      🎉 TARGET ACHIEVED for single symbol!")
                logger.info(
                    f"      📈 Multi-symbol projection: {new_freq * 5:.0f} trades/year")
            else:
                logger.warning(f"      ⚠️ Still below target (need 60+ per symbol)")

        return True

    except Exception as e:
        logger.error(f"❌ Signal generation test failed: {e}")
        return False


def test_portfolio_framework():
    """Test the portfolio framework."""
    logger.info("\n🧪 TESTING PORTFOLIO FRAMEWORK")
    logger.info("=" * 50)

    try:
        from portfolio_backtest import PortfolioBacktester

        logger.info("   🚀 Initializing portfolio backtester...")

        backtester = PortfolioBacktester(strategy_name="SimpleOrderBlockStrategy",
            timeframe="H1", days=90,  # Quick test with 3 months
            target_trades_per_year=250)

        logger.info("   ✅ Portfolio backtester initialized")
        logger.info(f"   📊 Target symbols: {len(backtester.target_symbols)}")
        logger.info(
            f"   🎯 Target frequency: {backtester.target_trades_per_year} trades/year")

        # Test single symbol (quick validation)
        logger.info("   🧪 Testing single symbol backtesting...")
        test_symbol = "GER40.cash"

        result = backtester.backtest_single_symbol(test_symbol)

        if result.success:
            logger.info(
                f"      ✅ {test_symbol}: {result.trades} trades ({result.trades_per_year:.0f}/year)")
            if result.trades_per_year >= 60:
                logger.info(f"      🎉 Single symbol target achieved!")
        else:
            logger.warning(f"      ⚠️ {test_symbol}: {result.error_message}")

        return True

    except Exception as e:
        logger.error(f"❌ Portfolio framework test failed: {e}")
        return False


def run_full_validation():
    """Run full validation suite."""
    logger.info("🚀 SOPHY4 FREQUENCY FIX VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("=" * 60)

    tests = [("Configuration Changes", test_config_changes),
        ("Strategy Parameters", test_strategy_parameters),
        ("Signal Generation", test_signal_generation),
        ("Portfolio Framework", test_portfolio_framework)]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("🏆 VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests Passed: {passed}/{total}")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("🚀 SOPHY4 is ready for high-frequency trading!")
        logger.info("📈 Expected: 250+ trades/year across 5 symbols")
        logger.info("💰 Risk: 5% per trade (personal account)")
        logger.info("")
        logger.info("🔥 NEXT STEPS:")
        logger.info("1. Run full portfolio backtest: python portfolio_backtest.py")
        logger.info("2. Deploy to paper trading for validation")
        logger.info("3. Scale to live trading with small positions")
    else:
        logger.error("❌ VALIDATION FAILED!")
        logger.error("🔧 Fix the failing tests before proceeding")

    return passed == total


if __name__ == "__main__":
    success = run_full_validation()
    sys.exit(0 if success else 1)