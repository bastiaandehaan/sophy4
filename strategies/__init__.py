# strategies/__init__.py
import importlib
import inspect
from pathlib import Path

from .base_strategy import BaseStrategy

# Globale dictionary voor geregistreerde strategieën
STRATEGIES = {}


def register_strategy(strategy_class):
    """
    Decorator om een strategie te registreren in het systeem.

    Gebruik:
        @register_strategy
        class MyStrategy(BaseStrategy):
            ...
    """
    if not inspect.isclass(strategy_class):
        raise TypeError("Decorator moet op een klasse worden toegepast")

    if not issubclass(strategy_class, BaseStrategy):
        raise TypeError(f"{strategy_class.__name__} moet BaseStrategy subklassen")

    # Registreer de strategie met zijn klassenaam
    STRATEGIES[strategy_class.__name__] = strategy_class
    print(f"Strategie '{strategy_class.__name__}' geregistreerd")

    return strategy_class


def get_strategy(strategy_name, **params):
    """
    Creëert een strategie-instantie op basis van de naam.

    Args:
        strategy_name: Naam van de strategie class
        **params: Parameters voor de strategie

    Returns:
        Een instantie van de gevraagde strategie
    """
    if strategy_name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(
            f"Strategie '{strategy_name}' niet gevonden. Beschikbaar: {available}")

    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(**params)


def list_strategies():
    """Toon alle beschikbare strategieën en hun parameters."""
    result = []
    for name, cls in STRATEGIES.items():
        params = cls.get_default_params()
        descriptions = cls.get_parameter_descriptions()
        result.append(
            {'name': name, 'parameters': params, 'descriptions': descriptions})
    return result


# Auto-import alle strategie modules om decorators uit te voeren
def _import_strategies():
    strategy_dir = Path(__file__).parent
    for file_path in strategy_dir.glob("*.py"):
        module_name = file_path.stem
        if module_name not in ['__init__', 'base_strategy']:
            try:
                importlib.import_module(f"strategies.{module_name}")
            except ImportError as e:
                print(f"Waarschuwing: Kon strategie '{module_name}' niet laden: {e}")


# Importeer alle strategieën bij import van deze module
_import_strategies()