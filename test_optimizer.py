# test_optimizer.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from optimization.optimize import optimize_strategy, validate_best_parameters
from strategies import get_strategy
import json
import time


def test_optimizer():
    """Test de optimize.py functionaliteit met een kleine parameter set."""
    print("=" * 80)
    print("OPTIMIZER TEST SCRIPT")
    print("=" * 80)

    # 1. Test met beperkte parameter set voor snelheid
    test_params = {'window': [20, 50, 80], 'std_dev': [1.5, 2.5],
        'sl_method': ['atr_based'], 'sl_atr_mult': [1.5, 2.5],
        'tp_method': ['atr_based'], 'tp_atr_mult': [3.0, 4.0]}

    start_time = time.time()
    print("\nTest 1: Beperkte parameter optimalisatie met BollongStrategy")
    results = optimize_strategy(strategy_name="BollongStrategy",
        param_ranges=test_params, metric="sharpe_ratio", top_n=2,
        ftmo_compliant_only=True, verbose=True)

    if not results or not results.get('top_results'):
        print("❌ Test 1 gefaald: Geen resultaten teruggekregen")
        return False

    print(f"✅ Test 1 geslaagd in {time.time() - start_time:.1f} seconden")

    # 2. Test andere metrics
    start_time = time.time()
    print("\nTest 2: Optimalisatie op Calmar ratio")
    calmar_results = optimize_strategy(strategy_name="BollongStrategy",
        param_ranges=test_params, metric="calmar_ratio", top_n=1,
        ftmo_compliant_only=True, verbose=True)

    if not calmar_results or not calmar_results.get('top_results'):
        print("❌ Test 2 gefaald: Geen resultaten teruggekregen")
        return False

    print(f"✅ Test 2 geslaagd in {time.time() - start_time:.1f} seconden")

    # 3. Test parameter opslag en laden
    best_params = results['top_results'][0]['params']

    # Sla parameters op
    with open('test_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    # Laad parameters en valideer
    with open('test_best_params.json', 'r') as f:
        loaded_params = json.load(f)

    print("\nTest 3: Parameter opslag en validatie")
    start_time = time.time()
    validation = validate_best_parameters(strategy_name="BollongStrategy",
        params=loaded_params)

    if not validation:
        print("❌ Test 3 gefaald: Validatie mislukt")
        return False

    print(f"✅ Test 3 geslaagd in {time.time() - start_time:.1f} seconden")

    # 4. Vergelijk met standaard parameters
    print("\nTest 4: Vergelijking met standaard parameters")
    # Maak strategie met standaard parameters
    default_strategy = get_strategy("BollongStrategy")
    # Maak strategie met geoptimaliseerde parameters
    optimized_strategy = get_strategy("BollongStrategy", **best_params)

    print(
        f"Standaard parameters: window={default_strategy.window}, std_dev={default_strategy.std_dev}")
    print(
        f"Geoptimaliseerde parameters: window={optimized_strategy.window}, std_dev={optimized_strategy.std_dev}")

    best_sharpe = results['top_results'][0]['metrics']['sharpe_ratio']
    print(f"Geoptimaliseerde Sharpe ratio: {best_sharpe:.2f}")

    print("\nALLE TESTS GESLAAGD!")
    return True


if __name__ == "__main__":
    test_optimizer()