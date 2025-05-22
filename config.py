# config.py - Centralized Configuration
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import MetaTrader5 as mt5


class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    initial_capital: float = 10000.0
    default_symbol: str = "GER40.cash"
    default_timeframe: str = "H1"
    default_lookback_days: int = 1095

    # Risk parameters
    max_risk_per_trade: float = 0.02
    max_daily_loss: float = 0.05
    max_total_loss: float = 0.10
    max_portfolio_heat: float = 0.06

    # Execution parameters
    fees: float = 0.00005
    slippage: float = 0.0001

    # FTMO compliance
    ftmo_daily_loss_limit: float = 0.05
    ftmo_total_loss_limit: float = 0.10
    ftmo_profit_target: float = 0.10


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    use_vectorbt: bool = True
    generate_plots: bool = True
    save_trades: bool = True
    save_results: bool = True
    plot_format: str = "png"  # png, html, both


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    default_trials: int = 100
    timeout_minutes: int = 30
    metric: str = "sharpe_ratio"
    use_walk_forward: bool = False
    walk_forward_windows: int = 5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5


class ConfigManager:
    """Centralized configuration manager."""

    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.base_dir = Path(__file__).parent
        self.config_dir = self.base_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        # Initialize configurations
        self.trading = TradingConfig()
        self.backtest = BacktestConfig()
        self.optimization = OptimizationConfig()
        self.logging_config = LoggingConfig()

        # Load configurations
        self._load_configs()
        self._setup_logging()
        self._load_symbols_and_timeframes()

    def _load_configs(self) -> None:
        """Load configuration from files."""
        config_files = {'trading': 'trading_config.json',
            'backtest': 'backtest_config.json',
            'optimization': 'optimization_config.json',
            'logging': 'logging_config.json'}

        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)
                        self._update_config(config_name, data)
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                # Create default config file
                self._save_default_config(config_name, config_path)

    def _update_config(self, config_name: str, data: Dict[str, Any]) -> None:
        """Update configuration with loaded data."""
        config_obj = getattr(self, config_name)
        for key, value in data.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)

    def _save_default_config(self, config_name: str, config_path: Path) -> None:
        """Save default configuration to file."""
        config_obj = getattr(self, config_name)
        with open(config_path, 'w') as f:
            json.dump(asdict(config_obj), f, indent=2)

    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set level
        level = getattr(logging, self.logging_config.level.upper())
        root_logger.setLevel(level)

        formatter = logging.Formatter(self.logging_config.format)

        # Console handler
        if self.logging_config.console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handler with rotation
        if self.logging_config.file_enabled:
            from logging.handlers import RotatingFileHandler

            log_file = log_dir / f"sophy4_{self.environment.value}.log"
            file_handler = RotatingFileHandler(log_file,
                maxBytes=self.logging_config.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging_config.backup_count)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def _load_symbols_and_timeframes(self) -> None:
        """Load symbols and timeframe configurations."""
        # Load timeframe config
        timeframe_config_path = self.base_dir / "timeframe_config.json"
        if timeframe_config_path.exists():
            with open(timeframe_config_path, 'r') as f:
                self.timeframe_config = json.load(f)
        else:
            self.timeframe_config = self._get_default_timeframe_config()

        # Load symbols
        self.symbols = self._get_available_symbols()
        self.pip_values = self._calculate_pip_values()

    def _get_default_timeframe_config(self) -> Dict[str, Any]:
        """Get default timeframe configuration."""
        return {"H1": {"freq": "1h", "window_range": [15, 20, 30, 50],
            "std_dev_range": [1.75, 2.0, 2.5],
            "sl_fixed_percent_range": [0.01, 0.015, 0.02, 0.025],
            "tp_fixed_percent_range": [0.03, 0.04, 0.05, 0.06],
            "risk_per_trade_range": [0.005, 0.0075, 0.01]},
            "D1": {"freq": "1d", "window_range": [20, 30, 40, 50],
                "std_dev_range": [2.0, 2.5, 3.0],
                "sl_fixed_percent_range": [0.02, 0.03, 0.04, 0.05],
                "tp_fixed_percent_range": [0.05, 0.06, 0.08, 0.1],
                "risk_per_trade_range": [0.005, 0.0075, 0.01]}}

    def _get_available_symbols(self) -> list:
        """Get available trading symbols."""
        default_symbols = ["EURUSD", "GER40.cash", "US30.cash", "XAUUSD"]

        # Try to get symbols from MT5
        try:
            if mt5.initialize():
                symbols_total = mt5.symbols_total()
                if symbols_total > 0:
                    # Get popular symbols that are visible
                    mt5_symbols = []
                    for symbol in default_symbols:
                        if mt5.symbol_info(symbol):
                            mt5_symbols.append(symbol)

                    if mt5_symbols:
                        return mt5_symbols
                mt5.shutdown()
        except Exception as e:
            logger.warning(f"Could not load symbols from MT5: {e}")

        return default_symbols

    def _calculate_pip_values(self) -> Dict[str, float]:
        """Calculate pip values for symbols."""
        pip_values = {}

        for symbol in self.symbols:
            try:
                if mt5.initialize():
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info:
                        tick_size = getattr(symbol_info, 'tick_size', 0.1)
                        pip_value = tick_size * symbol_info.trade_contract_size
                        pip_values[symbol] = pip_value
                    else:
                        pip_values[symbol] = 10.0  # Default
                mt5.shutdown()
            except Exception:
                pip_values[symbol] = 10.0  # Default

        return pip_values

    def get_strategy_params(self, strategy_name: str, timeframe: str) -> Dict[str, Any]:
        """Get strategy parameters for given timeframe."""
        tf_config = self.timeframe_config.get(timeframe,
                                              self.timeframe_config.get("H1", {}))

        # Merge with trading config
        params = {'symbol': self.trading.default_symbol,
            'risk_per_trade': self.trading.max_risk_per_trade,
            'max_daily_loss': self.trading.max_daily_loss,
            'max_total_loss': self.trading.max_total_loss, **tf_config}

        return params

    def get_output_dir(self) -> Path:
        """Get output directory for results."""
        output_dir = self.base_dir / "results" / self.environment.value
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    def save_config(self) -> None:
        """Save current configuration to files."""
        configs = {'trading': self.trading, 'backtest': self.backtest,
            'optimization': self.optimization, 'logging': self.logging_config}

        for config_name, config_obj in configs.items():
            config_path = self.config_dir / f"{config_name}_config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(config_obj), f, indent=2)

        logger.info("Configuration saved to files")

    def reload_config(self) -> None:
        """Reload configuration from files."""
        self._load_configs()
        self._setup_logging()
        logger.info("Configuration reloaded")


# Global configuration instance
config_manager = ConfigManager()

# Backward compatibility - expose commonly used values
logger = logging.getLogger(__name__)
INITIAL_CAPITAL = config_manager.trading.initial_capital
FEES = config_manager.trading.fees
SYMBOLS = config_manager.symbols
PIP_VALUES = config_manager.pip_values
OUTPUT_DIR = config_manager.get_output_dir()

# FTMO limits
MAX_DAILY_LOSS = config_manager.trading.ftmo_daily_loss_limit
MAX_TOTAL_LOSS = config_manager.trading.ftmo_total_loss_limit
PROFIT_TARGET = config_manager.trading.ftmo_profit_target


def get_pip_value(symbol: str) -> float:
    """Get pip value for symbol."""
    return PIP_VALUES.get(symbol, 10.0)


def get_strategy_config(strategy_name: str, timeframe: str) -> Dict[str, Any]:
    """Get configuration for strategy and timeframe."""
    return config_manager.get_strategy_params(strategy_name, timeframe)