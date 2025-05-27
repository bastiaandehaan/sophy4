"""
Strategies Module - PRODUCTION VERSION
Fixed: Proper parameter passing, strategy factory, Windows compatibility
Ensures: All strategy parameters are correctly applied from config system
"""
import logging
from typing import Dict, Any, Type, Optional
from pathlib import Path
import sys

# Windows-compatible logging
logger = logging.getLogger(__name__)

# Import available strategies
try:
    from .simple_order_block import SimpleOrderBlockStrategy
    from .base_strategy import BaseStrategy

    # Add other strategies here as they are developed
    # from .bollinger_strategy import BollingerStrategy
    # from .lstm_strategy import OrderBlockLSTMStrategy

except ImportError as e:
    logger.error(f"Failed to import strategies: {e}")
    sys.exit(1)

# Strategy registry for factory pattern
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "SimpleOrderBlockStrategy": SimpleOrderBlockStrategy,
    "simple_order_block": SimpleOrderBlockStrategy,  # Alternative name
    "order_block": SimpleOrderBlockStrategy,         # Short name

    # Add other strategies here
    # "BollingerStrategy": BollingerStrategy,
    # "OrderBlockLSTMStrategy": OrderBlockLSTMStrategy,
}

def get_available_strategies() -> Dict[str, str]:
    """
    Get list of available strategies with descriptions.

    Returns:
        Dict mapping strategy names to descriptions
    """
    descriptions = {
        "SimpleOrderBlockStrategy": "Order Block strategy optimized for frequency (400-600 trades/year)",
        # Add other strategy descriptions here
    }

    return {name: descriptions.get(name, "No description available")
            for name in STRATEGY_REGISTRY.keys()
            if not name.startswith("_")}  # Exclude internal aliases

def get_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """
    Factory function to create strategy instances with proper parameter passing.

    FIXED: Ensures all parameters from config system are properly applied.

    Args:
        strategy_name: Name of strategy to create
        **kwargs: Strategy parameters (from config system or overrides)

    Returns:
        Configured strategy instance

    Raises:
        ValueError: If strategy name not found
        TypeError: If strategy creation fails
    """
    logger.info(f"Creating {strategy_name} with params: {list(kwargs.keys())}")

    # Normalize strategy name
    strategy_name = strategy_name.strip()

    # Find strategy class
    strategy_class = None
    for registered_name, cls in STRATEGY_REGISTRY.items():
        if registered_name.lower() == strategy_name.lower():
            strategy_class = cls
            break

    if strategy_class is None:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")

    try:
        # Validate required parameters
        symbol = kwargs.get('symbol')
        if not symbol:
            logger.warning("No symbol specified, using default 'GER40.cash'")
            kwargs['symbol'] = 'GER40.cash'

        trading_mode = kwargs.get('trading_mode', 'personal')
        logger.info(f"Creating {strategy_name} for {symbol} in {trading_mode.upper()} mode")

        # Log key parameters for verification
        key_params = ['use_htf_confirmation', 'stress_threshold', 'rsi_min', 'rsi_max', 'risk_per_trade']
        for param in key_params:
            if param in kwargs:
                logger.info(f"  {param}: {kwargs[param]}")

        # Create strategy instance with ALL parameters
        # CRITICAL: Ensure **kwargs are passed to preserve all config parameters
        strategy = strategy_class(**kwargs)

        # Verify strategy was created with correct parameters
        if hasattr(strategy, 'get_strategy_info'):
            info = strategy.get_strategy_info()
            logger.info(f"Strategy created successfully: {info.get('name', strategy_name)}")

            # Verify critical parameters were applied
            if 'use_htf_confirmation' in kwargs:
                actual_htf = getattr(strategy, 'use_htf_confirmation', None)
                expected_htf = kwargs['use_htf_confirmation']
                if actual_htf != expected_htf:
                    logger.warning(f"HTF confirmation mismatch: expected {expected_htf}, got {actual_htf}")
                else:
                    logger.info(f"  HTF confirmation correctly set: {actual_htf}")

        return strategy

    except TypeError as e:
        logger.error(f"Strategy creation failed - invalid parameters: {e}")
        logger.error(f"Strategy class: {strategy_class}")
        logger.error(f"Parameters passed: {kwargs}")
        raise TypeError(f"Failed to create {strategy_name}: {e}")

    except Exception as e:
        logger.error(f"Unexpected error creating {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        raise

def validate_strategy_parameters(strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize strategy parameters.

    Args:
        strategy_name: Name of strategy
        params: Parameters dictionary

    Returns:
        Validated parameters dictionary

    Raises:
        ValueError: If parameters are invalid
    """
    validated_params = params.copy()

    # Common parameter validation
    if 'symbol' in validated_params:
        symbol = validated_params['symbol']
        if not isinstance(symbol, str) or len(symbol) == 0:
            raise ValueError(f"Invalid symbol: {symbol}")

    if 'risk_per_trade' in validated_params:
        risk = validated_params['risk_per_trade']
        if not isinstance(risk, (int, float)) or risk <= 0 or risk > 1:
            raise ValueError(f"Invalid risk_per_trade: {risk} (must be between 0 and 1)")

    if 'rsi_min' in validated_params and 'rsi_max' in validated_params:
        rsi_min = validated_params['rsi_min']
        rsi_max = validated_params['rsi_max']
        if rsi_min >= rsi_max:
            raise ValueError(f"Invalid RSI range: {rsi_min}-{rsi_max} (min must be < max)")
        if rsi_min < 0 or rsi_max > 100:
            raise ValueError(f"Invalid RSI values: {rsi_min}-{rsi_max} (must be 0-100)")

    # Strategy-specific validation
    if strategy_name.lower() in ['simpleorderblockstrategy', 'simple_order_block', 'order_block']:
        # Validate order block specific parameters
        if 'ob_lookback' in validated_params:
            lookback = validated_params['ob_lookback']
            if not isinstance(lookback, int) or lookback < 1 or lookback > 20:
                raise ValueError(f"Invalid ob_lookback: {lookback} (must be 1-20)")

        if 'stress_threshold' in validated_params:
            threshold = validated_params['stress_threshold']
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                raise ValueError(f"Invalid stress_threshold: {threshold} (must be positive)")

        if 'min_wick_ratio' in validated_params:
            ratio = validated_params['min_wick_ratio']
            if not isinstance(ratio, (int, float)) or ratio < 0:
                raise ValueError(f"Invalid min_wick_ratio: {ratio} (must be non-negative)")

    logger.info(f"Parameters validated for {strategy_name}")
    return validated_params

def list_strategy_parameters(strategy_name: str) -> Dict[str, Any]:
    """
    Get parameter information for a strategy.

    Args:
        strategy_name: Name of strategy

    Returns:
        Dictionary with parameter info
    """
    if strategy_name.lower() in ['simpleorderblockstrategy', 'simple_order_block', 'order_block']:
        return {
            'required_parameters': ['symbol'],
            'optional_parameters': {
                'trading_mode': 'personal',
                'use_htf_confirmation': False,
                'stress_threshold': 10.0,
                'min_wick_ratio': 0.001,
                'use_rejection_wicks': False,
                'use_session_filter': False,
                'use_volume_filter': False,
                'rsi_min': 1,
                'rsi_max': 99,
                'rsi_period': 14,
                'volume_multiplier': 0.1,
                'volume_period': 20,
                'risk_per_trade': 0.05,
                'ob_lookback': 2,
                'sl_percent': 0.008,
                'tp_percent': 0.025,
                'min_body_ratio': 0.5,
                'trend_strength_min': 0.5,
                'htf_timeframe': 'H4',
                'htf_lookback': 30,
            },
            'parameter_descriptions': {
                'symbol': 'Trading symbol (e.g., GER40.cash)',
                'trading_mode': 'Trading mode (personal/ftmo/aggressive)',
                'use_htf_confirmation': 'Enable higher timeframe confirmation',
                'stress_threshold': 'Market stress detection threshold',
                'min_wick_ratio': 'Minimum wick ratio requirement',
                'rsi_min': 'Minimum RSI value for signals',
                'rsi_max': 'Maximum RSI value for signals',
                'risk_per_trade': 'Risk percentage per trade',
                'ob_lookback': 'Order block lookback periods',
                'sl_percent': 'Stop loss percentage',
                'tp_percent': 'Take profit percentage',
            }
        }

    return {"error": f"Parameter info not available for {strategy_name}"}

def create_strategy_from_config(config_manager, strategy_name: str, symbol: str, **overrides) -> BaseStrategy:
    """
    Create strategy using config manager with optional overrides.

    Args:
        config_manager: ConfigManager instance
        strategy_name: Strategy to create
        symbol: Trading symbol
        **overrides: Parameter overrides

    Returns:
        Configured strategy instance
    """
    # Get parameters from config
    params = config_manager.get_strategy_params(strategy_name, symbol, **overrides)

    # Validate parameters
    validated_params = validate_strategy_parameters(strategy_name, params)

    # Create strategy
    return get_strategy(strategy_name, **validated_params)

# Module initialization
logger.info(f"Strategies module loaded: {len(STRATEGY_REGISTRY)} strategies available")
available_strategies = get_available_strategies()
for name, desc in available_strategies.items():
    if not any(alias in name for alias in ['simple_order_block', 'order_block']):  # Only log main names
        logger.info(f"  {name}: {desc}")

# Export public interface
__all__ = [
    'get_strategy',
    'get_available_strategies',
    'validate_strategy_parameters',
    'list_strategy_parameters',
    'create_strategy_from_config',
    'SimpleOrderBlockStrategy',
    'BaseStrategy',
    'STRATEGY_REGISTRY'
]