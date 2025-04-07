# test_extended_strategy.py
from strategies import get_strategy


def test_extended_params():
    # Test met standaard parameters
    strategy1 = get_strategy("BollongStrategy")
    print(f"Strategie geïnstantieerd met standaard parameters: {strategy1.name}")

    # Test met aangepaste parameters
    strategy2 = get_strategy("BollongStrategy", window=40, std_dev=2.5,
        sl_method="fixed_percent", sl_fixed_percent=0.025, tp_method="atr_based",
        tp_atr_mult=4.0, use_trailing_stop=True, trailing_stop_percent=0.02)
    print(f"Strategie geïnstantieerd met aangepaste parameters:")
    print(f"  Window: {strategy2.window}")
    print(f"  Stop-loss: {strategy2.sl_method} ({strategy2.sl_fixed_percent})")
    print(f"  Trailing stop: {strategy2.use_trailing_stop}")

    try:
        strategy2.validate_parameters()
        print("Parameter validatie geslaagd")
    except ValueError as e:
        print(f"Parameter validatie mislukt: {e}")

    return True


if __name__ == "__main__":
    test_extended_params()