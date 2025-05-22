# utils/parameter_manager.py - Unified Parameter Loading System
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from pydantic import BaseModel, validator, Field
import pandas as pd

logger = logging.getLogger(__name__)


class ParameterValidationError(Exception):
    """Raised when parameter validation fails."""
    pass


class ParameterConfig(BaseModel):
    """Pydantic model for parameter validation."""

    # Core strategy parameters
    symbol: str = Field(..., regex=r'^[A-Z0-9._]+$', description="Trading symbol")
    timeframe: str = Field(..., regex=r'^(M[1-9]|H[1-4]|D1|W1|MN1)$',
                           description="Chart timeframe")

    # Risk management
    risk_per_trade: float = Field(ge=0.001, le=0.1,
                                  description="Risk per trade (0.1-10%)")
    max_daily_loss: float = Field(ge=0.01, le=0.2, description="Max daily loss (1-20%)")
    max_total_loss: float = Field(ge=0.05, le=0.5, description="Max total loss (5-50%)")

    # Position sizing
    initial_capital: float = Field(ge=1000, le=1000000, description="Initial capital")
    position_multiplier: float = Field(ge=0.1, le=5.0,
                                       description="Position size multiplier")

    # Strategy-specific (optional)
    sl_percent: Optional[float] = Field(None, ge=0.001, le=0.1,
                                        description="Stop loss percentage")
    tp_percent: Optional[float] = Field(None, ge=0.001, le=0.5,
                                        description="Take profit percentage")

    # Indicator parameters
    bb_window: Optional[int] = Field(None, ge=5, le=200,
                                     description="Bollinger Bands window")
    bb_std_dev: Optional[float] = Field(None, ge=0.5, le=5.0,
                                        description="Bollinger Bands std dev")
    rsi_period: Optional[int] = Field(None, ge=5, le=50, description="RSI period")

    # Order block parameters
    ob_lookback: Optional[int] = Field(None, ge=1, le=50,
                                       description="Order block lookback")
    ob_strength: Optional[float] = Field(None, ge=1.0, le=10.0,
                                         description="Order block strength")

    # LSTM parameters
    lstm_threshold: Optional[float] = Field(None, ge=0.0, le=1.0,
                                            description="LSTM threshold")
    model_confidence: Optional[float] = Field(None, ge=0.0, le=1.0,
                                              description="Model confidence threshold")

    @validator('symbol')
    def symbol_must_be_valid(cls, v):
        """Validate symbol format."""
        if not v or len(v) < 3:
            raise ValueError('Symbol must be at least 3 characters')
        return v.upper()

    @validator('timeframe')
    def timeframe_must_be_valid(cls, v):
        """Validate timeframe format."""
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
        if v not in valid_timeframes:
            raise ValueError(f'Timeframe must be one of {valid_timeframes}')
        return v


@dataclass
class ParameterSet:
    """Single parameter set with metadata."""
    name: str
    description: str
    parameters: Dict[str, Any]
    source: str  # file, optimization, default
    timestamp: str
    validation_status: str  # valid, invalid, unknown
    performance_metrics: Optional[Dict[str, float]] = None


class UnifiedParameterManager:
    """
    Unified parameter loading and management system.

    Features:
    - Load parameters from multiple sources
    - Validate parameters with Pydantic
    - Merge parameter sets with precedence rules
    - Export optimized parameters
    - Version control for parameter changes
    """

    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path("config")
        self.config_dir.mkdir(exist_ok=True)

        # Parameter storage
        self.parameter_sets: Dict[str, ParameterSet] = {}
        self.default_parameters: Dict[str, Dict] = {}

        # Load order (precedence from low to high)
        self.load_sources = ['defaults',  # Built-in defaults
            'config_files',  # JSON config files
            'optimization',  # Optimization results
            'user_override'  # User-specified overrides
        ]

        self._initialize_defaults()
        logger.info("Unified Parameter Manager initialized")

    def _initialize_defaults(self) -> None:
        """Initialize default parameters for each strategy."""

        # SimpleOrderBlockStrategy defaults
        self.default_parameters['SimpleOrderBlockStrategy'] = {
            'H1': {'symbol': 'GER40.cash', 'timeframe': 'H1', 'risk_per_trade': 0.01,
                'ob_lookback': 5, 'sl_percent': 0.01, 'tp_percent': 0.03,
                'rsi_period': 14, 'rsi_min': 35, 'rsi_max': 65,
                'volume_multiplier': 1.2, 'volume_period': 20, 'bb_window': 20,
                'bb_std_dev': 1.5, 'dynamic_sizing': True},
            'D1': {'symbol': 'GER40.cash', 'timeframe': 'D1', 'risk_per_trade': 0.01,
                'ob_lookback': 10, 'sl_percent': 0.02, 'tp_percent': 0.05,
                'rsi_period': 14, 'rsi_min': 30, 'rsi_max': 70,
                'volume_multiplier': 1.3, 'volume_period': 20, 'bb_window': 30,
                'bb_std_dev': 2.0, 'dynamic_sizing': True}}

        # OrderBlockLSTMStrategy defaults
        self.default_parameters['OrderBlockLSTMStrategy'] = {
            'H1': {'symbol': 'GER40.cash', 'timeframe': 'H1', 'risk_per_trade': 0.0075,
                'ob_lookback': 20, 'ob_strength': 2.0, 'sl_fixed_percent': 0.02,
                'tp_fixed_percent': 0.045, 'lstm_threshold': 0.4,
                'model_confidence': 0.6, 'use_trailing_stop': True,
                'trailing_stop_percent': 0.01}}

        # BollongStrategy defaults
        self.default_parameters['BollongStrategy'] = {
            'H1': {'symbol': 'EURUSD', 'timeframe': 'H1', 'risk_per_trade': 0.01,
                'window': 20, 'std_dev': 2.0, 'sl_fixed_percent': 0.015,
                'tp_fixed_percent': 0.03, 'use_trailing_stop': True,
                'trailing_stop_percent': 0.01}}

    def load_parameters(self, strategy_name: str, timeframe: str = "H1",
                        source_file: Optional[Path] = None,
                        optimization_file: Optional[Path] = None,
                        user_overrides: Optional[Dict] = None) -> ParameterConfig:
        """
        Load and merge parameters from multiple sources.

        Args:
            strategy_name: Name of the strategy
            timeframe: Chart timeframe
            source_file: Optional JSON file with parameters
            optimization_file: Optional optimization results file
            user_overrides: Optional user-specified parameter overrides

        Returns:
            Validated ParameterConfig object

        Raises:
            ParameterValidationError: If parameters are invalid
        """
        logger.info(f"Loading parameters for {strategy_name} on {timeframe}")

        # Start with defaults
        merged_params = self._get_default_parameters(strategy_name, timeframe)

        # Apply config file parameters
        if source_file and source_file.exists():
            file_params = self._load_from_file(source_file)
            merged_params.update(file_params)
            logger.info(f"Applied parameters from {source_file}")

        # Apply optimization results
        if optimization_file and optimization_file.exists():
            opt_params = self._load_optimization_results(optimization_file)
            merged_params.update(opt_params)
            logger.info(f"Applied optimization results from {optimization_file}")

        # Apply user overrides (highest precedence)
        if user_overrides:
            merged_params.update(user_overrides)
            logger.info(f"Applied user overrides: {list(user_overrides.keys())}")

        # Validate parameters
        try:
            validated_params = ParameterConfig(**merged_params)
            logger.info("✅ Parameter validation successful")
            return validated_params

        except Exception as e:
            error_msg = f"Parameter validation failed: {str(e)}"
            logger.error(error_msg)
            raise ParameterValidationError(error_msg)

    def _get_default_parameters(self, strategy_name: str, timeframe: str) -> Dict[
        str, Any]:
        """Get default parameters for strategy and timeframe."""
        strategy_defaults = self.default_parameters.get(strategy_name, {})
        timeframe_defaults = strategy_defaults.get(timeframe,
                                                   strategy_defaults.get('H1', {}))

        if not timeframe_defaults:
            logger.warning(
                f"No defaults found for {strategy_name}, using generic defaults")
            return {'symbol': 'GER40.cash', 'timeframe': timeframe,
                'risk_per_trade': 0.01, 'initial_capital': 10000,
                'max_daily_loss': 0.05, 'max_total_loss': 0.10,
                'position_multiplier': 1.0}

        return timeframe_defaults.copy()

    def _load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Load parameters from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle different file formats
            if isinstance(data, list):
                # Optimization results format
                return data[0].get('params', {}) if data else {}
            elif isinstance(data, dict):
                # Direct parameter format
                return data.get('params', data)
            else:
                logger.warning(f"Unknown parameter file format: {type(data)}")
                return {}

        except Exception as e:
            logger.error(f"Failed to load parameters from {file_path}: {e}")
            return {}

    def _load_optimization_results(self, file_path: Path, index: int = 0) -> Dict[
        str, Any]:
        """Load parameters from optimization results file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if isinstance(data, list) and len(data) > index:
                params = data[index].get('params', {})
                logger.info(f"Loaded optimization result #{index} from {file_path}")
                return params
            else:
                logger.warning(
                    f"Invalid optimization file format or index: {file_path}")
                return {}

        except Exception as e:
            logger.error(f"Failed to load optimization results from {file_path}: {e}")
            return {}

    def save_parameters(self, strategy_name: str, timeframe: str,
                        parameters: ParameterConfig, description: str = "",
                        performance_metrics: Optional[Dict[str, float]] = None) -> Path:
        """Save parameters with metadata."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_{timeframe}_params_{timestamp}.json"
        file_path = self.config_dir / filename

        # Create parameter set
        param_set = ParameterSet(name=f"{strategy_name}_{timeframe}",
            description=description, parameters=parameters.dict(), source="user_saved",
            timestamp=timestamp, validation_status="valid",
            performance_metrics=performance_metrics)

        # Save to file
        with open(file_path, 'w') as f:
            json.dump(asdict(param_set), f, indent=2)

        logger.info(f"Parameters saved to {file_path}")
        return file_path

    def get_optimization_grid(self, strategy_name: str, timeframe: str) -> Dict[
        str, List[Any]]:
        """Get parameter grid for optimization."""
        base_params = self._get_default_parameters(strategy_name, timeframe)

        # Define optimization ranges based on strategy
        if strategy_name == "SimpleOrderBlockStrategy":
            return {'ob_lookback': [3, 5, 7, 10],
                'sl_percent': [0.008, 0.01, 0.012, 0.015, 0.02],
                'tp_percent': [0.025, 0.03, 0.035, 0.04, 0.05],
                'rsi_period': [10, 14, 18], 'rsi_min': [30, 35, 40],
                'rsi_max': [60, 65, 70], 'volume_multiplier': [1.1, 1.2, 1.3, 1.5],
                'bb_window': [15, 20, 25], 'bb_std_dev': [1.0, 1.5, 2.0],
                'risk_per_trade': [0.005, 0.0075, 0.01, 0.0125]}
        elif strategy_name == "OrderBlockLSTMStrategy":
            return {'ob_lookback': [15, 20, 25], 'ob_strength': [1.5, 2.0, 2.5, 3.0],
                'lstm_threshold': [0.3, 0.4, 0.5, 0.6],
                'sl_fixed_percent': [0.015, 0.02, 0.025],
                'tp_fixed_percent': [0.03, 0.04, 0.05],
                'model_confidence': [0.5, 0.6, 0.7],
                'risk_per_trade': [0.005, 0.0075, 0.01]}
        else:
            # Generic optimization grid
            return {'risk_per_trade': [0.005, 0.01, 0.015],
                'sl_percent': [0.01, 0.015, 0.02], 'tp_percent': [0.03, 0.04, 0.05]}

    def validate_parameter_compatibility(self, strategy_name: str,
                                         parameters: Dict[str, Any]) -> List[str]:
        """Validate parameter compatibility with strategy."""
        warnings = []

        # Check strategy-specific requirements
        if strategy_name == "SimpleOrderBlockStrategy":
            if parameters.get('ob_lookback', 0) > 20:
                warnings.append("Large ob_lookback may reduce signal frequency")

            if parameters.get('rsi_min', 0) >= parameters.get('rsi_max', 100):
                warnings.append("RSI min should be less than RSI max")

        elif strategy_name == "OrderBlockLSTMStrategy":
            if parameters.get('lstm_threshold', 0) > 0.8:
                warnings.append(
                    "High LSTM threshold may reduce signal frequency significantly")

        # General risk warnings
        if parameters.get('risk_per_trade', 0) > 0.02:
            warnings.append("Risk per trade >2% may violate FTMO rules")

        if parameters.get('sl_percent', 0) > parameters.get('tp_percent', 0):
            warnings.append(
                "Stop loss larger than take profit creates poor risk/reward")

        return warnings

    def create_parameter_template(self, strategy_name: str, timeframe: str) -> Path:
        """Create a parameter template file for manual editing."""
        template_params = self._get_default_parameters(strategy_name, timeframe)
        grid = self.get_optimization_grid(strategy_name, timeframe)

        template = {"strategy": strategy_name, "timeframe": timeframe,
            "description": f"Parameter template for {strategy_name} on {timeframe}",
            "parameters": template_params, "optimization_ranges": grid,
            "notes": ["Edit the 'parameters' section with your desired values",
                "Use 'optimization_ranges' as a guide for reasonable values",
                "All percentage values are in decimal form (0.01 = 1%)",
                "Save this file and load with --params argument"]}

        template_path = self.config_dir / f"{strategy_name}_{timeframe}_template.json"
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)

        logger.info(f"Parameter template created: {template_path}")
        return template_path


# Global parameter manager instance
parameter_manager = UnifiedParameterManager()


def load_strategy_parameters(strategy_name: str, timeframe: str = "H1",
                             params_file: Optional[Path] = None,
                             optimization_file: Optional[Path] = None,
                             user_overrides: Optional[Dict] = None) -> ParameterConfig:
    """
    Convenience function to load strategy parameters.

    Args:
        strategy_name: Name of the strategy
        timeframe: Chart timeframe
        params_file: Optional parameter file
        optimization_file: Optional optimization results
        user_overrides: Optional parameter overrides

    Returns:
        Validated parameter configuration
    """
    return parameter_manager.load_parameters(strategy_name=strategy_name,
        timeframe=timeframe, source_file=params_file,
        optimization_file=optimization_file, user_overrides=user_overrides)


def create_parameter_template(strategy_name: str, timeframe: str = "H1") -> Path:
    """Create parameter template file."""
    return parameter_manager.create_parameter_template(strategy_name, timeframe)


def get_optimization_grid(strategy_name: str, timeframe: str = "H1") -> Dict[
    str, List[Any]]:
    """Get optimization parameter grid."""
    return parameter_manager.get_optimization_grid(strategy_name, timeframe)