#!/usr/bin/env python3
"""
🔥 IMMEDIATE SOPHY4 PARAMETER FIX TEST
Test if we can force the right parameters and get signals
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def test_parameter_forcing():
    """Test if forcing parameters bypasses the config issues."""
    print("=== SOPHY4 IMMEDIATE PARAMETER TEST ===")
    print("Testing parameter forcing to bypass config issues...")

    try:
        # Import the strategy directly
        from strategies.simple_order_block import SimpleOrderBlockStrategy
        from backtest.data_loader import fetch_historical_data

        print("✅ Modules imported successfully")

        # Get test data (3 months for speed)
        print("📥 Loading test data...")
        df = fetch_historical_data("GER40.cash", "H1", days=90)

        if df is None or df.empty:
            print("❌ CRITICAL: No data available")
            return False

        print(f"✅ Data loaded: {len(df)} bars")

        # FORCE the frequency-optimized parameters
        print("\n🔧 Testing FORCED frequency parameters...")

        forced_params = {# 🔥 CRITICAL FIXES
            'use_htf_confirmation': False,  # DISABLE HTF BLOCKING
            'stress_threshold': 4.0,  # RELAX vs 2.2
            'min_wick_ratio': 0.05,  # MINIMAL vs 0.3
            'use_rejection_wicks': False,  # DISABLE WICKS
            'use_session_filter': False,  # 24/7 TRADING

            # Wide RSI range
            'rsi_min': 5,  # vs 25 FTMO
            'rsi_max': 95,  # vs 75 FTMO

            # Relaxed volume
            'volume_multiplier': 0.8,  # vs 1.1 FTMO
            'use_volume_filter': False,  # OPTIONAL

            # Core params
            'symbol': 'GER40.cash', 'ob_lookback': 5, 'sl_percent': 0.01,
            'tp_percent': 0.03, 'risk_per_trade': 0.05  # 5% PERSONAL vs 1.5% FTMO
        }

        print("🎯 Creating strategy with FORCED parameters:")
        for key, value in forced_params.items():
            if 'use_' in key or 'threshold' in key or 'ratio' in key:
                print(f"   {key}: {value}")

        # Create strategy with forced params
        strategy = SimpleOrderBlockStrategy(**forced_params)

        # VERIFY the parameters stuck
        print("\n🔍 VERIFICATION - Checking if parameters were applied:")
        actual_htf = getattr(strategy, 'use_htf_confirmation', 'MISSING')
        actual_stress = getattr(strategy, 'stress_threshold', 'MISSING')
        actual_wick = getattr(strategy, 'min_wick_ratio', 'MISSING')
        actual_rsi_min = getattr(strategy, 'rsi_min', 'MISSING')

        print(f"   HTF Confirmation: {actual_htf} (should be False)")
        print(f"   Stress Threshold: {actual_stress} (should be 4.0)")
        print(f"   Min Wick Ratio: {actual_wick} (should be 0.05)")
        print(f"   RSI Min: {actual_rsi_min} (should be 5)")

        # Generate signals
        print("\n🔄 Generating signals...")
        entries, sl, tp = strategy.generate_signals(df)

        signals = entries.sum()
        trades_per_year = signals * (365 / len(df))

        print(f"\n📊 RESULTS:")
        print(f"   Signals Generated: {signals}")
        print(f"   Projected Trades/Year: {trades_per_year:.0f}")

        # Success criteria
        if trades_per_year >= 60:
            print(f"\n🎉 SUCCESS! {trades_per_year:.0f} trades/year per symbol")
            print(f"💰 Multi-symbol projection: {trades_per_year * 5:.0f} trades/year")
            print("🚀 SOLUTION CONFIRMED: Parameter forcing works!")
            return True
        elif trades_per_year > 20:
            print(
                f"\n⚠️ PROGRESS: {trades_per_year:.0f} trades/year (improved but still low)")
            print("🔧 Need more aggressive parameters or longer test period")
            return False
        else:
            print(f"\n❌ FAILED: Still only {trades_per_year:.0f} trades/year")
            print("🚨 HTF confirmation may still be blocking signals")
            return False

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_parameter_forcing()

    print("\n" + "=" * 60)
    if success:
        print("✅ PARAMETER FORCING WORKS!")
        print("🚀 Next: Run full portfolio test")
        print("💡 Command: python portfolio_backtest.py")
    else:
        print("❌ ISSUE: Parameters not taking effect")
        print("🔧 Check HTF confirmation logic in strategy")
        print("🔍 May need to hardcode HTF return True")
    print("=" * 60)