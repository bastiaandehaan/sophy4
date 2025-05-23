# strategies/__init__.py - Streamlined & Centralized
import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .base_strategy import BaseStrategy

# Global strategy registry
STRATEGIES = {}

logger = logging.getLogger(__name__)


def register_strategy(strategy_class):
    """
    Clean decorator to register strategies in the system.
    """
    if not inspect.isclass(strategy_class):
        raise TypeError("Decorator must be applied to a class")
    if not issubclass(strategy_class, BaseStrategy):
        raise TypeError(f"{strategy_class.__name__} must inherit from BaseStrategy")

    STRATEGIES[strategy_class.__name__] = strategy_class

    if os.getenv('SOPHY4_DEBUG'):
        logger.debug(f"Strategy '{strategy_class.__name__}' registered")

    return strategy_class


def get_strategy(strategy_name: str, **params) -> BaseStrategy:
    """
    Create strategy instance with centralized configuration.

    Args:
        strategy_name: Name of the strategy class
        **params: Override parameters (symbol, timeframe, etc.)

    Returns:
        Configured strategy instance
    """
    # Get strategy class with case-insensitive matching
    strategy_class = _find_strategy_class(strategy_name)

    # Import config manager here to avoid circular imports
    from config import config_manager

    # Get base configuration from centralized system
    timeframe = params.get('timeframe', 'H1')
    base_config = config_manager.get_strategy_params(strategy_name, timeframe)

    # Merge base config with user overrides
    final_params = {**base_config, **params}

    # Filter parameters to only include what the strategy accepts
    valid_params = _filter_valid_parameters(strategy_class, final_params)

    logger.info(f"Creating {strategy_name} with params: {list(valid_params.keys())}")

    return strategy_class(**valid_params)


def _find_strategy_class(strategy_name: str):
    """Find strategy class with case-insensitive matching."""
    # Try exact match first
    if strategy_name in STRATEGIES:
        return STRATEGIES[strategy_name]

    # Try case-insensitive match
    strategy_name_lower = strategy_name.lower()
    matches = [name for name in STRATEGIES.keys() if
               name.lower() == strategy_name_lower]

    if len(matches) == 1:
        return STRATEGIES[matches[0]]
    elif len(matches) > 1:
        logger.warning(
            f"Multiple strategies found for '{strategy_name}', using '{matches[0]}'")
        return STRATEGIES[matches[0]]
    else:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(
            f"Strategy '{strategy_name}' not found. Available: {available}")


def _filter_valid_parameters(strategy_class, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter parameters to only include what the strategy constructor accepts."""
    init_signature = inspect.signature(strategy_class.__init__)
    valid_params = {}

    for param_name, param_value in params.items():
        if param_name in init_signature.parameters:
            valid_params[param_name] = param_value
        else:
            logger.debug(
                f"Parameter '{param_name}' ignored for {strategy_class.__name__}")

    return valid_params


def list_strategies() -> list:
    """List all available strategies with their accepted parameters."""
    result = []

    for name, cls in STRATEGIES.items():
        # Get accepted parameters from __init__ signature
        init_params = list(inspect.signature(cls.__init__).parameters.keys())
        if 'self' in init_params:
            init_params.remove('self')

        # Get strategy info without relying on strategy-level methods
        strategy_info = {'name': name, 'accepted_params': init_params,
            'module': cls.__module__, 'description': cls.__doc__.split('\n')[
                0] if cls.__doc__ else "No description"}

        result.append(strategy_info)

    return result


def get_strategy_names() -> list:
    """Get list of all registered strategy names."""
    return list(STRATEGIES.keys())


def is_strategy_available(strategy_name: str) -> bool:
    """Check if a strategy is available."""
    return strategy_name in STRATEGIES or any(
        name.lower() == strategy_name.lower() for name in STRATEGIES.keys())


# Auto-import all strategy modules
def _import_strategies():
    """Import all strategy modules to trigger @register_strategy decorators."""
    strategy_dir = Path(__file__).parent
    excluded_files = {'__init__', 'base_strategy',
                      'bollong_vectorized'}  # Exclude deleted files

    for file_path in strategy_dir.glob("*.py"):
        module_name = file_path.stem

        if module_name not in excluded_files:
            try:
                importlib.import_module(f"strategies.{module_name}")
                logger.debug(f"Imported strategy module: {module_name}")
            except ImportError as e:
                logger.warning(f"Failed to import strategy '{module_name}': {e}")

    logger.info(f"Loaded {len(STRATEGIES)} strategies: {', '.join(STRATEGIES.keys())}")


# Import all strategies when module is loaded
_import_strategies()