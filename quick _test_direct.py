#!/usr/bin/env python3
"""
Direct Strategy Test - Bypass broken backtest parameter passing
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def test_strategy_direct():
    """Test strategy directly without backtest framework"""

    print("🚀 DIRECT STRATEGY TEST (Bypassing Backtest)")
    print("=" * 60)

    try:
        from strategies.simple_order_block import SimpleOrderBlockStrategy
        from backtest.data_loader import fetch_historical_data

        # Haal data op
        print("📥 Loading data for GER40.cash H1...")
        df = fetch_historical_data("GER40.cash", "H1", days=365)

        if df is None or df.empty:
            print("❌ Failed to load data")
            return

        print(f"✅ Data loaded: {len(df)} bars")

        # Test configurations
        configs = [("CURRENT DEFAULT", {}),  # Use strategy defaults

            ("RESTRICTIVE", {'use_htf_confirmation': True, 'stress_threshold': 2.2,
                'min_wick_ratio': 0.3, 'use_rejection_wicks': True, 'rsi_min': 25,
                'rsi_max': 75}),

            ("RELAXED", {'use_htf_confirmation': False,  # DISABLE HTF blocking
                'stress_threshold': 3.5,  # RELAX stress filter
                'min_wick_ratio': 0.15,  # LOWER wick requirements
                'use_rejection_wicks': True,  # Keep wick filter
                'rsi_min': 15,  # WIDER RSI range
                'rsi_max': 85}),

            ("AGGRESSIVE", {'use_htf_confirmation': False,  # DISABLE HTF blocking
                'stress_threshold': 4.0,  # VERY RELAXED stress
                'min_wick_ratio': 0.05,  # MINIMAL wick requirements
                'use_rejection_wicks': False,  # DISABLE wick filter completely
                'use_session_filter': False,  # 24/7 trading
                'rsi_min': 5,  # VERY WIDE RSI range
                'rsi_max': 95, 'volume_multiplier': 0.8  # RELAXED volume filter
            })]

        results = []

        for config_name, params in configs:
            print(f"\n🧪 Testing {config_name} configuration...")
            print(f"   Parameters: {params}")

            try:
                # Create strategy with parameters
                strategy = SimpleOrderBlockStrategy(**params)

                # Show key parameters
                print(f"   HTF Confirmation: {strategy.use_htf_confirmation}")
                print(
                    f"   Stress Threshold: {getattr(strategy, 'stress_threshold', 'DEFAULT')}")
                print(
                    f"   Min Wick Ratio: {getattr(strategy, 'min_wick_ratio', 'DEFAULT')}")
                print(
                    f"   Rejection Wicks: {getattr(strategy, 'use_rejection_wicks', 'DEFAULT')}")

                # Generate signals
                entries, sl, tp = strategy.generate_signals(df)

                # Calculate results
                total_signals = entries.sum()
                trades_per_year = total_signals * (365 / len(df))

                print(
                    f"   📊 RESULT: {total_signals} signals = {trades_per_year:.0f} trades/year")

                results.append({'name': config_name, 'signals': total_signals,
                    'trades_per_year': trades_per_year, 'params': params})

            except Exception as e:
                print(f"   ❌ FAILED: {str(e)}")
                import traceback
                traceback.print_exc()

        # Compare results
        print(f"\n📊 RESULTS COMPARISON")
        print("=" * 60)
        print(
            f"{'Configuration':<15} {'Signals':<8} {'Trades/Year':<12} {'vs Default':<10}")
        print("-" * 60)

        baseline = None
        for result in results:
            if result['name'] == 'CURRENT DEFAULT':
                baseline = result['trades_per_year']

            multiplier = result[
                             'trades_per_year'] / baseline if baseline and baseline > 0 else 1

            print(
                f"{result['name']:<15} {result['signals']:<8} {result['trades_per_year']:<12.0f} {multiplier:<10.1f}x")

        # Analysis
        print(f"\n🎯 ANALYSIS")
        print("=" * 30)

        best_result = max(results, key=lambda x: x['trades_per_year'])

        print(f"🏆 BEST CONFIG: {best_result['name']}")
        print(f"   Trades/Year: {best_result['trades_per_year']:.0f}")

        if best_result['trades_per_year'] >= 60:
            print(f"   ✅ TARGET ACHIEVED: 60+ trades per symbol")
            print(
                f"   💡 Multi-symbol projection: {best_result['trades_per_year'] * 5:.0f} trades/year across 5 symbols")
        elif best_result['trades_per_year'] >= 30:
            print(f"   ⚠️  GOOD PROGRESS: 30+ trades per symbol")
            print(f"   💡 Try even more aggressive parameters")
        else:
            print(f"   ❌ STILL TOO LOW: Need more parameter relaxation")

        # Success criteria
        target_met = best_result['trades_per_year'] >= 60

        if target_met:
            print(f"\n🎉 SUCCESS! Parameter control is working")
            print(f"🔧 Next step: Fix backtest parameter passing")
            print(
                f"📈 Expected portfolio frequency: {best_result['trades_per_year'] * 5:.0f} trades/year")
        else:
            print(f"\n⚠️  Parameters working but need more aggressive settings")
            print(f"💡 Try disabling more filters or using wider ranges")

        return results

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    print("🔬 SOPHY4 DIRECT STRATEGY TEST")
    print("This bypasses the broken backtest parameter passing")
    print("=" * 60)

    results = test_strategy_direct()

    if results:
        print(f"\n✅ TEST COMPLETED - Parameters are working!")
    else:
        print(f"\n❌ TEST FAILED - Check error messages above")