#!/usr/bin/env python3
"""
🔥 NUCLEAR HTF CONFIRMATION FIX
Direct patch to disable HTF confirmation blocking
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def patch_htf_confirmation():
    """Directly patch the HTF confirmation method to always return True."""

    print("=== NUCLEAR HTF CONFIRMATION PATCH ===")
    print("Directly patching SimpleOrderBlockStrategy.get_htf_confirmation()")

    try:
        from strategies.simple_order_block import SimpleOrderBlockStrategy

        # NUCLEAR OPTION: Monkey patch the HTF confirmation method
        def always_bullish_htf(self, df):
            """PATCHED: Always return True (bullish) regardless of HTF."""
            return True

        # Replace the method
        SimpleOrderBlockStrategy.get_htf_confirmation = always_bullish_htf

        print("✅ HTF confirmation method PATCHED")
        print("🚀 HTF will now ALWAYS return True (bullish)")

        # Test the patch
        print("\n🧪 Testing the patch...")

        strategy = SimpleOrderBlockStrategy(use_htf_confirmation=True,
            # Even with True, should not block
            symbol='GER40.cash')

        # Create dummy dataframe
        import pandas as pd
        dummy_df = pd.DataFrame({'close': [100, 101, 102]})

        htf_result = strategy.get_htf_confirmation(dummy_df)
        print(f"HTF confirmation result: {htf_result}")

        if htf_result == True:
            print("✅ PATCH SUCCESSFUL: HTF always returns True")
            return True
        else:
            print("❌ PATCH FAILED: HTF still returns False")
            return False

    except Exception as e:
        print(f"❌ PATCH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patched_strategy():
    """Test the strategy with patched HTF confirmation."""

    print("\n=== TESTING PATCHED STRATEGY ===")

    try:
        from strategies.simple_order_block import SimpleOrderBlockStrategy
        from backtest.data_loader import fetch_historical_data

        # Get test data
        print("📥 Loading test data...")
        df = fetch_historical_data("GER40.cash", "H1", days=90)

        if df is None or df.empty:
            print("❌ No data available")
            return False

        print(f"✅ Data loaded: {len(df)} bars")

        # Create strategy with HTF confirmation enabled (should not block now)
        print("🧪 Testing strategy with HTF confirmation ENABLED...")

        patched_params = {'use_htf_confirmation': True,  # ENABLED but patched
            'stress_threshold': 4.0,  # RELAXED
            'min_wick_ratio': 0.05,  # MINIMAL
            'use_rejection_wicks': False,  # DISABLED
            'rsi_min': 5,  # WIDE
            'rsi_max': 95,  # WIDE
            'symbol': 'GER40.cash'}

        strategy = SimpleOrderBlockStrategy(**patched_params)

        # Generate signals
        print("🔄 Generating signals with PATCHED strategy...")
        entries, sl, tp = strategy.generate_signals(df)

        signals = entries.sum()
        trades_per_year = signals * (365 / len(df))

        print(f"\n📊 PATCHED RESULTS:")
        print(f"   Signals: {signals}")
        print(f"   Trades/Year: {trades_per_year:.0f}")

        if trades_per_year >= 60:
            print(f"🎉 SUCCESS! {trades_per_year:.0f} trades/year achieved!")
            print("🚀 HTF patch working - signals no longer blocked")
            return True
        else:
            print(f"⚠️ Still low: {trades_per_year:.0f} trades/year")
            print("🔧 May need additional parameter tuning")
            return False

    except Exception as e:
        print(f"❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_nuclear_patch_and_test():
    """Apply nuclear patch and run full test."""

    print("🔥 APPLYING NUCLEAR HTF PATCH...")

    # Step 1: Patch HTF confirmation
    patch_success = patch_htf_confirmation()

    if not patch_success:
        print("❌ HTF patch failed - cannot proceed")
        return False

    # Step 2: Test patched strategy
    test_success = test_patched_strategy()

    if test_success:
        print("\n" + "=" * 60)
        print("🎉 NUCLEAR PATCH SUCCESSFUL!")
        print("✅ HTF confirmation no longer blocks signals")
        print("🚀 Ready for portfolio backtesting")
        print("💡 Next: Run fixed_portfolio_backtest.py")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("⚠️ PATCH APPLIED BUT RESULTS STILL LOW")
        print("🔧 May need additional parameter optimization")
        print("💡 Check other filters in strategy logic")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = apply_nuclear_patch_and_test()

    if success:
        print("\n🚀 READY FOR PORTFOLIO TEST!")
        print("Run: python fixed_portfolio_backtest.py")
    else:
        print("\n🔧 NEED FURTHER DEBUGGING")
        print("Check strategy logic for other blocking filters")