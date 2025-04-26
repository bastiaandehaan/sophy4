# strategies/__init__.py
import importlib
import inspect
from pathlib import Path
import logging

from .base_strategy import BaseStrategy

# Globale dictionary voor geregistreerde strategieën
STRATEGIES = {}

# Configureer een lokale logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def register_strategy(strategy_class):
    """
    Decorator om een strategie te registreren in het systeem.
    """
    if not inspect.isclass(strategy_class):
        raise TypeError("Decorator moet op een klasse worden toegepast")
    if not issubclass(strategy_class, BaseStrategy):
        raise TypeError(f"{strategy_class.__name__} moet BaseStrategy subklassen")

    STRATEGIES[strategy_class.__name__] = strategy_class
    print(f"Strategie '{strategy_class.__name__}' geregistreerd")
    return strategy_class


def get_strategy(strategy_name, **params):
    """
    Creëert een strategie-instantie op basis van de naam.
    Verbeterd met case-insensitieve matching en foutafhandeling.
    """
    # Probeer exacte match
    if strategy_name in STRATEGIES:
        strategy_class = STRATEGIES[strategy_name]
    else:
        # Probeer case-insensitieve match
        strategy_name_lower = strategy_name.lower()
        matched_strategies = [name for name in STRATEGIES.keys() if
                              name.lower() == strategy_name_lower]

        if len(matched_strategies) == 1:
            # Eén match gevonden
            strategy_name = matched_strategies[0]
            strategy_class = STRATEGIES[strategy_name]
            logger.info(
                f"Strategie '{strategy_name}' gevonden via case-insensitieve match voor '{strategy_name}'")
        elif len(matched_strategies) > 1:
            # Meerdere matches, neem de eerste en waarschuw
            strategy_name = matched_strategies[0]
            strategy_class = STRATEGIES[strategy_name]
            logger.warning(
                f"Meerdere strategieën gevonden voor '{strategy_name}', gebruik '{strategy_name}'")
        else:
            # Geen match, toon beschikbare strategieën
            available = ", ".join(STRATEGIES.keys())
            raise ValueError(
                f"Strategie '{strategy_name}' niet gevonden. Beschikbaar: {available}")

    # Haal de signature op van de __init__ methode
    init_signature = inspect.signature(strategy_class.__init__)
    valid_params = {}

    # Filter de parameters om alleen geldige door te geven
    for param_name, param_value in params.items():
        if param_name in init_signature.parameters:
            valid_params[param_name] = param_value
        else:
            logger.warning(
                f"Parameter '{param_name}' wordt genegeerd voor strategie {strategy_name}, "
                f"niet aanwezig in __init__ signature")

    # Log welke parameters we gebruiken
    logger.info(
        f"Strategie {strategy_name} wordt geïnitialiseerd met parameters: {valid_params}")

    # Maak een instantie met alleen de geldige parameters
    return strategy_class(**valid_params)


def list_strategies():
    """Toon alle beschikbare strategieën en hun parameters."""
    result = []
    for name, cls in STRATEGIES.items():
        params = cls.get_default_params()
        descriptions = cls.get_parameter_descriptions()

        # Voeg ook informatie toe over welke parameters de __init__ methode accepteert
        init_params = list(inspect.signature(cls.__init__).parameters.keys())
        # Verwijder 'self' uit de lijst
        if 'self' in init_params:
            init_params.remove('self')

        result.append({'name': name, 'parameters': params, 'descriptions': descriptions,
            'accepted_params': init_params})
    return result


def debug_strategies():
    """Helper functie om geregistreerde strategieën te debuggen."""
    strategy_names = ", ".join(STRATEGIES.keys())
    logger.info(f"Geregistreerde strategieën: {strategy_names}")
    for name, cls in STRATEGIES.items():
        logger.info(f"  - {name}: {cls.__module__}.{cls.__name__}")


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

    # Log de geregistreerde strategieën voor debug doeleinden
    debug_strategies()


# Importeer alle strategieën bij import van deze module
_import_strategies()