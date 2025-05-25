#!/usr/bin/env python3
"""
QUICK FIX SCRIPT for Sophy4 Frequency Issue
Run this to immediately test if the parameter forcing works
"""


def quick_frequency_test():
    """Quick test to see if we can force the right parameters."""

    print("=== SOPHY4 QUICK FREQUENCY FIX TEST ===")
    print("Testing if we can force frequency-optimized parameters")
    print("=" * 60)

    try:
        import sys
        from pathlib import Path

        # Ensure we can import the modules
        sys.path.append(str(Path(__file__).parent))

        from strategies.simple_order_block import SimpleOrderBlockStrategy
        from backtest.data_loader import fetch_historical_data

        print("1. Modules imported successfully")

        # Get test data
        print("2. Loading test data...")
        df = fetch_historical_data("GER40.cash", "H1", days=90)  # 3 months

        if df is None or df.empty:
            print("   ERROR: Could not load data")
            return False

        print(f"   SUCCESS: {len(df)} bars loaded")

        # Test OLD vs NEW parameters side by side
        print("\n3. Testing parameter configurations...")

        configs = [("OLD (FTMO Style)", {'use_htf_confirmation': True,  # BLOCKING
            'stress_threshold': 2.2,  # RESTRICTIVE
            'min_wick_ratio': 0.3,  # STRICT
            'use_rejection_wicks': True,  # REQUIRED
            'rsi_min': 25, 'rsi_max': 75,  # NARROW
            'symbol': 'GER40.cash'}),

            ("NEW (Frequency Optimized)", {'use_htf_confirmation': False,  # ALLOW ALL
                'stress_threshold': 4.0,  # RELAXED
                'min_wick_ratio': 0.05,  # MINIMAL
                'use_rejection_wicks': False,  # OPTIONAL
                'rsi_min': 5, 'rsi_max': 95,  # WIDE
                'symbol': 'GER40.cash'})]

        results = []

        for name, params in configs:
            print(f"\n   Testing {name}...")
            print(f"     HTF Confirmation: {params['use_htf_confirmation']}")
            print(f"     Stress Threshold: {params['stress_threshold']}")
            print(f"     Min Wick Ratio: {params['min_wick_ratio']}")

            try:
                # Create strategy with forced parameters
                strategy = SimpleOrderBlockStrategy(**params)

                # Verify parameters were applied
                actual_htf = getattr(strategy, 'use_htf_confirmation', 'Unknown')
                actual_stress = getattr(strategy, 'stress_threshold', 'Unknown')
                actual_wick = getattr(strategy, 'min_wick_ratio', 'Unknown')

                print(f"     VERIFIED HTF: {actual_htf}")
                print(f"     VERIFIED Stress: {actual_stress}")
                print(f"     VERIFIED Wick: {actual_wick}")

                # Generate signals
                entries, sl, tp = strategy.generate_signals(df)
                signals = entries.sum()
                trades_per_year = signals * (365 / len(df))

                print(
                    f"     RESULT: {signals} signals = {trades_per_year:.0f} trades/year")

                results.append((name, signals, trades_per_year))

            except Exception as e:
                print(f"     ERROR: {str(e)}")
                results.append((name, 0, 0))

        # Compare results
        print("\n4. COMPARISON RESULTS:")
        print("-" * 60)

        for name, signals, trades_year in results:
            status = "SUCCESS" if trades_year >= 60 else "TOO LOW"
            print(
                f"{name:<25} {signals:>6} signals {trades_year:>6.0f}/year [{status}]")

        # Analysis
        print("\n5. ANALYSIS:")

        if len(results) >= 2:
            old_freq = results[0][2]
            new_freq = results[1][2]

            if new_freq > old_freq:
                improvement = new_freq / max(old_freq, 1)
                print(f"   IMPROVEMENT: {improvement:.1f}x increase in frequency")

                if new_freq >= 60:
                    print("   TARGET ACHIEVED: 60+ trades per symbol!")
                    print(f"   Multi-symbol projection: {new_freq * 5:.0f} trades/year")
                    print("   SOLUTION CONFIRMED: Parameter forcing works!")
                    return True
                else:
                    print(f"   PROGRESS: Increased but still below 60 target")
                    print("   ACTION: Need even more aggressive parameters")
                    return False
            else:
                print("   PROBLEM: No improvement in frequency")
                print("   ISSUE: Parameters may not be applied correctly")
                return False
        else:
            print("   ERROR: Could not complete comparison")
            return False

    except ImportError as e:
        print(f"   IMPORT ERROR: {str(e)}")
        print("   SOLUTION: Ensure you're in the Sophy4 project directory")
        return False
    except Exception as e:
        print(f"   UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def show_fix_instructions(success):
    """Show next steps based on test results."""

    print("\n" + "=" * 60)
    print("=== NEXT STEPS ===")

    if success:
        print("SUCCESS! The parameter forcing works.")
        print("\nTo fix your portfolio backtester:")
        print(
            "1. Replace your portfolio_backtest.py with the Windows-compatible version")
        print("2. The new version forces the correct parameters directly")
        print("3. Run: python portfolio_backtest.py")
        print("4. Expected result: 250+ trades/year across 5 symbols")

    else:
        print("ISSUE: Parameter forcing didn't achieve the target.")
        print("\nPossible causes:")
        print("1. HTF confirmation filter still blocking signals")
        print("2. Market conditions are genuinely not favorable")
        print("3. Strategy logic needs adjustment")

        print("\nImmediate actions:")
        print("1. Check if HTF confirmation is truly disabled")
        print("2. Try the super aggressive parameters")
        print("3. Test with different time periods or symbols")

    print("\nFor more help:")
    print("- Review the strategy logs for 'HTF bearish - blocking all signals'")
    print("- Check that 'use_htf_confirmation=False' is actually applied")
    print("- Test with a longer data period (365 days instead of 90)")


if __name__ == "__main__":
    result = quick_frequency_test()
    show_fix_instructions(result)