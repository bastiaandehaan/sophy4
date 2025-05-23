# config.py - Unified Configuration System
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import MetaTrader5 as mt5


class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class SymbolConfig:
    """Symbol-specific configuration."""
    symbol: str
    pip_value: float = 10.0
    contract_size: float = 1.0
    tick_size: float = 0.1
    volume_min: float = 0.01
    volume_max: float = 100.0
    spread_typical: float = 0.0001


@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    initial_capital: float = 10000.0
    default_symbol: str = "GER40.cash"
    default_timeframe: str = "H1"
    default_lookback_days: int = 1095

    # Risk parameters - SINGLE SOURCE OF TRUTH
    max_risk_per_trade: float = 0.01  # 1% base risk
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
class StrategyDefaults:
    """Default strategy parameters by timeframe."""
    timeframe: str

    # Bollinger Bands defaults
    bb_window_range: List[int] = field(default_factory=lambda: [20, 30, 50])
    bb_std_dev_range: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])

    # Risk management defaults
    sl_percent_range: List[float] = field(default_factory=lambda: [0.01, 0.015, 0.02])
    tp_percent_range: List[float] = field(default_factory=lambda: [0.03, 0.04, 0.05])

    # Order block defaults
    ob_lookback_range: List[int] = field(default_factory=lambda: [5, 10, 15])
    ob_strength_range: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])

    # LSTM defaults
    lstm_threshold_range: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5])

    # Risk defaults
    risk_per_trade_range: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.015])

    # Trailing stop defaults
    trailing_stop_range: List[float] = field(
        default_factory=lambda: [0.01, 0.015, 0.02])


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    use_vectorbt: bool = True
    generate_plots: bool = True
    save_trades: bool = True
    save_results: bool = True
    plot_format: str = "png"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5


class UnifiedConfigManager:
    """Centralized configuration manager - SINGLE SOURCE OF TRUTH."""

    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.base_dir = Path(__file__).parent
        self.config_dir = self.base_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        # Initialize configurations
        self.trading = TradingConfig()
        self.backtest = BacktestConfig()
        self.logging_config = LoggingConfig()

        # Initialize symbol configurations
        self.symbols: Dict[str, SymbolConfig] = {}

        # Initialize strategy defaults by timeframe
        self.strategy_defaults: Dict[str, StrategyDefaults] = {}

        # Load all configurations
        self._load_configs()
        self._setup_logging()
        self._initialize_symbols()
        self._initialize_strategy_defaults()

    def _load_configs(self) -> None:
        """Load configuration from files."""
        config_files = {'trading': 'trading_config.json',
            'backtest': 'backtest_config.json', 'logging_config': 'logging_config.json'}

        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)
                        self._update_config(config_name, data)
                except Exception as e:
                    print(
                        f"Warning: Failed to load {filename}: {e}")  # Use print instead of logger
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

    def _initialize_symbols(self) -> None:
        """Initialize symbol configurations."""
        default_symbols = ["EURUSD", "GER40.cash", "US30.cash", "XAUUSD", "GBPUSD"]

        # Try to get symbols from MT5
        mt5_symbols = self._fetch_mt5_symbols(default_symbols)

        # Create symbol configurations
        for symbol in mt5_symbols:
            symbol_info = self._get_mt5_symbol_info(symbol)
            if symbol_info:
                self.symbols[symbol] = SymbolConfig(symbol=symbol,
                    pip_value=symbol_info.get('pip_value', 10.0),
                    contract_size=symbol_info.get('contract_size', 1.0),
                    tick_size=symbol_info.get('tick_size', 0.1),
                    volume_min=symbol_info.get('volume_min', 0.01),
                    volume_max=symbol_info.get('volume_max', 100.0))
            else:
                # Fallback configuration
                self.symbols[symbol] = SymbolConfig(symbol=symbol)

    def _fetch_mt5_symbols(self, default_symbols: List[str]) -> List[str]:
        """Fetch available symbols from MT5."""
        try:
            if mt5.initialize():
                available_symbols = []
                for symbol in default_symbols:
                    if mt5.symbol_info(symbol):
                        available_symbols.append(symbol)
                mt5.shutdown()
                return available_symbols if available_symbols else default_symbols
        except Exception as e:
            print(
                f"Warning: Could not load symbols from MT5: {e}")  # Use print instead of logger

        return default_symbols

    def _get_mt5_symbol_info(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get symbol information from MT5."""
        try:
            if mt5.initialize():
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    result = {
                        'pip_value': getattr(symbol_info, 'trade_tick_value', 10.0),
                        'contract_size': symbol_info.trade_contract_size,
                        'tick_size': symbol_info.tick_size,
                        'volume_min': symbol_info.volume_min,
                        'volume_max': symbol_info.volume_max}
                    mt5.shutdown()
                    return result
                mt5.shutdown()
        except Exception as e:
            print(
                f"Warning: Error getting symbol info for {symbol}: {e}")  # Use print instead of logger

        return None

    def _initialize_strategy_defaults(self) -> None:
        """Initialize strategy defaults by timeframe."""
        # M5 defaults
        self.strategy_defaults["M5"] = StrategyDefaults(timeframe="M5",
            bb_window_range=[5, 10, 15, 20], bb_std_dev_range=[1.0, 1.5, 2.0],
            sl_percent_range=[0.005, 0.01, 0.015], tp_percent_range=[0.01, 0.02, 0.03],
            ob_lookback_range=[3, 5, 10], lstm_threshold_range=[0.2, 0.3, 0.4],
            risk_per_trade_range=[0.005, 0.01],
            trailing_stop_range=[0.005, 0.01, 0.015])

        # H1 defaults
        self.strategy_defaults["H1"] = StrategyDefaults(timeframe="H1",
            bb_window_range=[15, 20, 30, 50], bb_std_dev_range=[1.75, 2.0, 2.5],
            sl_percent_range=[0.01, 0.015, 0.02, 0.025],
            tp_percent_range=[0.03, 0.04, 0.05, 0.06],
            ob_lookback_range=[5, 10, 15, 20], ob_strength_range=[1.5, 2.0, 2.5],
            lstm_threshold_range=[0.35, 0.4, 0.45, 0.5],
            risk_per_trade_range=[0.005, 0.0075, 0.01],
            trailing_stop_range=[0.008, 0.01, 0.015])

        # D1 defaults
        self.strategy_defaults["D1"] = StrategyDefaults(timeframe="D1",
            bb_window_range=[20, 30, 40, 50, 60], bb_std_dev_range=[2.0, 2.5, 3.0],
            sl_percent_range=[0.02, 0.03, 0.04, 0.05],
            tp_percent_range=[0.05, 0.06, 0.08, 0.1],
            ob_lookback_range=[10, 15, 20, 25], ob_strength_range=[2.0, 2.5, 3.0],
            lstm_threshold_range=[0.4, 0.5, 0.6],
            risk_per_trade_range=[0.005, 0.0075, 0.01],
            trailing_stop_range=[0.015, 0.02, 0.025])

    # UNIFIED API METHODS
    def get_symbol_config(self, symbol: str) -> SymbolConfig:
        """Get symbol configuration."""
        return self.symbols.get(symbol, SymbolConfig(symbol=symbol))

    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        return self.get_symbol_config(symbol).pip_value

    def get_strategy_params(self, strategy_name: str, timeframe: str) -> Dict[str, Any]:
        """Get strategy parameters for given timeframe - SINGLE SOURCE."""
        defaults = self.strategy_defaults.get(timeframe,
                                              self.strategy_defaults.get("H1"))
        symbol_config = self.get_symbol_config(self.trading.default_symbol)

        # UNIFIED parameter set
        params = {# Base trading config
            'symbol': self.trading.default_symbol,
            'initial_capital': self.trading.initial_capital,
            'risk_per_trade': self.trading.max_risk_per_trade,
            'max_daily_loss': self.trading.max_daily_loss,
            'max_total_loss': self.trading.max_total_loss,

            # Symbol-specific
            'pip_value': symbol_config.pip_value,
            'contract_size': symbol_config.contract_size,

            # Strategy defaults from timeframe
            'bb_window_range': defaults.bb_window_range,
            'bb_std_dev_range': defaults.bb_std_dev_range,
            'sl_percent_range': defaults.sl_percent_range,
            'tp_percent_range': defaults.tp_percent_range,
            'ob_lookback_range': defaults.ob_lookback_range,
            'ob_strength_range': defaults.ob_strength_range,
            'lstm_threshold_range': defaults.lstm_threshold_range,
            'trailing_stop_range': defaults.trailing_stop_range,

            # Execution
            'fees': self.trading.fees, 'slippage': self.trading.slippage}

        return params

    def get_output_dir(self) -> Path:
        """Get output directory for results."""
        output_dir = self.base_dir / "results" / self.environment.value
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return list(self.symbols.keys())

    def save_config(self) -> None:
        """Save current configuration to files."""
        configs = {'trading': self.trading, 'backtest': self.backtest,
            'logging_config': self.logging_config}

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


# Global configuration instance - SINGLE SOURCE OF TRUTH
config_manager = UnifiedConfigManager()

# Create logger AFTER config_manager is initialized
logger = logging.getLogger(__name__)

# Backward compatibility exports - SINGLE SOURCE
INITIAL_CAPITAL = config_manager.trading.initial_capital
FEES = config_manager.trading.fees
SYMBOLS = config_manager.get_available_symbols()
OUTPUT_DIR = config_manager.get_output_dir()

# FTMO limits - SINGLE SOURCE
MAX_DAILY_LOSS = config_manager.trading.ftmo_daily_loss_limit
MAX_TOTAL_LOSS = config_manager.trading.ftmo_total_loss_limit
PROFIT_TARGET = config_manager.trading.ftmo_profit_target


# UNIFIED API FUNCTIONS
def get_pip_value(symbol: str) -> float:
    """Get pip value for symbol - SINGLE SOURCE."""
    return config_manager.get_pip_value(symbol)


def get_symbol_info(symbol: str) -> SymbolConfig:
    """Get complete symbol information."""
    return config_manager.get_symbol_config(symbol)


def get_strategy_config(strategy_name: str, timeframe: str) -> Dict[str, Any]:
    """Get configuration for strategy and timeframe - SINGLE SOURCE."""
    return config_manager.get_strategy_params(strategy_name, timeframe)


def get_risk_config() -> Dict[str, float]:
    """Get risk management configuration."""
    return {'max_risk_per_trade': config_manager.trading.max_risk_per_trade,
        'max_daily_loss': config_manager.trading.max_daily_loss,
        'max_total_loss': config_manager.trading.max_total_loss,
        'max_portfolio_heat': config_manager.trading.max_portfolio_heat}