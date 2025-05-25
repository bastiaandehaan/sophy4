#!/usr/bin/env python3
"""
Backtest Parameter Passing Fix
The strategy constructor now works, but parameters aren't reaching the backtest
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def debug_backtest_parameter_flow():
    """Debug where parameters get lost in the backtest chain."""

    print("🔍 DEBUGGING BACKTEST PARAMETER FLOW")
    print("=" * 50)

    # Test the full chain: run_extended_backtest -> get_strategy -> strategy
    print("\n1. Testing direct strategy creation (known working)...")

    test_params = {'use_htf_confirmation': False, 'stress_threshold': 3.5,
        'symbol': 'GER40.cash'}

    try:
        from strategies.simple_order_block import SimpleOrderBlockStrategy
        strategy = SimpleOrderBlockStrategy(**test_params)
        print(f"   ✅ Direct creation works: HTF={strategy.use_htf_confirmation}")
    except Exception as e:
        print(f"   ❌ Direct creation failed: {e}")
        return False

    print("\n2. Testing get_strategy function...")
    try:
        from strategies import get_strategy
        strategy = get_strategy("SimpleOrderBlockStrategy", **test_params)
        print(f"   ✅ get_strategy works: HTF={strategy.use_htf_confirmation}")
    except Exception as e:
        print(f"   ❌ get_strategy failed: {e}")
        return False

    print("\n3. Testing run_extended_backtest parameter passing...")
    try:
        from backtest.backtest import run_extended_backtest

        # This should pass parameters to the strategy
        print(f"   Calling run_extended_backtest with parameters...")
        pf, metrics = run_extended_backtest(strategy_name="SimpleOrderBlockStrategy",
            parameters=test_params, symbol="GER40.cash", timeframe="H1", period_days=30,
            # Short test
            silent=False  # See the logs
        )

        if metrics and 'trades_count' in metrics:
            print(f"   ✅ Backtest completed: {metrics['trades_count']} trades")
            return True
        else:
            print(f"   ❌ Backtest failed or no metrics")
            return False

    except Exception as e:
        print(f"   ❌ run_extended_backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_backtest_parameter_passing():
    """Analyze and fix the parameter passing in backtest.py"""

    print(f"\n🔧 ANALYZING BACKTEST.PY PARAMETER FLOW")
    print("=" * 50)

    # Check how run_extended_backtest calls get_strategy
    try:
        import inspect
        from backtest.backtest import run_extended_backtest

        # Get the source code
        source = inspect.getsource(run_extended_backtest)

        # Look for get_strategy call
        if "get_strategy" in source:
            print("✅ run_extended_backtest calls get_strategy")

            # Check how parameters are passed
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if 'get_strategy' in line:
                    print(f"   Line {i}: {line.strip()}")

                    # Check surrounding lines for parameter passing
                    for j in range(max(0, i - 2), min(len(lines), i + 3)):
                        if 'parameters' in lines[j]:
                            print(f"   Context {j}: {lines[j].strip()}")
        else:
            print("❌ run_extended_backtest doesn't call get_strategy!")
            print("This is the problem - parameters never reach the strategy")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def create_fixed_backtest_call():
    """Create a fixed version of the backtest call that properly passes parameters."""

    print(f"\n🚀 CREATING FIXED BACKTEST FUNCTION")
    print("=" * 50)

    fixed_function = '''
def run_fixed_backtest(strategy_name: str, parameters: Dict[str, Any], 
                      symbol: str, timeframe: str = "H1", days: int = 365):
    """
    Fixed backtest that properly passes parameters to strategy.
    """
    from datetime import datetime
    from backtest.data_loader import fetch_historical_data
    from strategies import get_strategy
    import vectorbt as vbt
    from config import INITIAL_CAPITAL, FEES

    print(f"🧪 FIXED BACKTEST: {strategy_name} with parameters")
    print(f"Parameters being passed: {parameters}")

    # Fetch data
    df = fetch_historical_data(symbol, timeframe=timeframe, days=days)
    if df is None or df.empty:
        print(f"❌ No data for {symbol}")
        return None, {}

    print(f"✅ Data loaded: {len(df)} bars")

    # Create strategy with parameters - THIS IS THE FIX!
    strategy = get_strategy(strategy_name, **parameters)
    print(f"✅ Strategy created with HTF={getattr(strategy, 'use_htf_confirmation', 'N/A')}")

    # Generate signals
    entries, sl_stop, tp_stop = strategy.generate_signals(df)

    print(f"✅ Signals generated: {entries.sum()} entries")

    # Create portfolio
    try:
        pf = vbt.Portfolio.from_signals(
            close=df['close'],
            entries=entries > 0,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            init_cash=INITIAL_CAPITAL,
            fees=FEES
        )

        # Calculate basic metrics
        metrics = {
            'trades_count': len(pf.trades) if hasattr(pf, 'trades') else 0,
            'total_return': float(pf.total_return()),
            'win_rate': float(pf.trades.win_rate()) if len(pf.trades) > 0 else 0.0
        }

        print(f"✅ Portfolio created: {metrics['trades_count']} trades")
        return pf, metrics

    except Exception as e:
        print(f"❌ Portfolio creation failed: {e}")
        return None, {}
'''

    # Write to file
    with open("fixed_backtest.py", "w", encoding='utf-8') as f:
        f.write("# Fixed Backtest Function\n")
        f.write("from typing import Dict, Any\n")
        f.write(fixed_function)

    print("✅ Created fixed_backtest.py")
    return fixed_function


def test_fixed_backtest():
    """Test the fixed backtest function."""

    print(f"\n🎯 TESTING FIXED BACKTEST")
    print("=" * 50)

    # Import the fixed function
    exec(open("fixed_backtest.py").read())

    # Test restrictive vs relaxed
    test_cases = [("RESTRICTIVE",
                   {'use_htf_confirmation': True, 'stress_threshold': 2.2,
                       'min_wick_ratio': 0.3, 'symbol': 'GER40.cash'}),
        ("RELAXED", {'use_htf_confirmation': False,  # KEY CHANGE
            'stress_threshold': 3.5,  # KEY CHANGE
            'min_wick_ratio': 0.1,  # KEY CHANGE
            'use_rejection_wicks': False,  # KEY CHANGE
            'symbol': 'GER40.cash'})]

    results = []

    for name, params in test_cases:
        print(f"\n🧪 Testing {name} configuration...")
        try:
            # Use the fixed function (defined in exec above)
            pf, metrics = run_fixed_backtest(strategy_name="SimpleOrderBlockStrategy",
                parameters=params, symbol="GER40.cash", timeframe="H1", days=365)

            if metrics:
                trades = metrics.get('trades_count', 0)
                win_rate = metrics.get('win_rate', 0)
                print(f"   ✅ {name}: {trades} trades, {win_rate:.1%} win rate")
                results.append((name, trades, win_rate))
            else:
                print(f"   ❌ {name}: No results")

        except Exception as e:
            print(f"   ❌ {name}: Failed - {e}")

    # Compare results
    if len(results) >= 2:
        restrictive_trades = results[0][1]
        relaxed_trades = results[1][1]

        print(f"\n📊 COMPARISON:")
        print(f"   Restrictive: {restrictive_trades} trades")
        print(f"   Relaxed: {relaxed_trades} trades")

        if relaxed_trades > restrictive_trades:
            multiplier = relaxed_trades / max(restrictive_trades, 1)
            print(f"   🚀 SUCCESS: {multiplier:.1f}x increase!")
            print(f"   💡 Multi-symbol projection: {relaxed_trades * 5} trades/year")
        else:
            print(f"   ⚠️  Need more aggressive parameter relaxation")


if __name__ == "__main__":
    print("🔧 BACKTEST PARAMETER PASSING DEBUG")
    print("=" * 60)

    # Step 1: Debug where parameters get lost
    backtest_works = debug_backtest_parameter_flow()

    # Step 2: Analyze the backtest code
    fix_backtest_parameter_passing()

    # Step 3: Create and test fixed version
    create_fixed_backtest_call()
    test_fixed_backtest()

    print(f"\n🎯 NEXT STEPS:")
    if not backtest_works:
        print(
            "1. The issue is in run_extended_backtest - parameters not passed to get_strategy")
        print("2. Use the fixed_backtest.py function to test parameters properly")
        print("3. Once working, fix the main backtest.py function")
    else:
        print("1. Parameters are working correctly!")
        print("2. Test with more aggressive parameter relaxation")