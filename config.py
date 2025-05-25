# config.py - UPDATED FOR PERSONAL TRADING (Not FTMO)
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
    """Trading-specific configuration - UPDATED FOR PERSONAL TRADING."""
    initial_capital: float = 100000.0  # Personal account size
    default_symbol: str = "GER40.cash"
    default_timeframe: str = "H1"
    default_lookback_days: int = 1095

    # 🚀 PERSONAL TRADING RISK (NOT FTMO)
    max_risk_per_trade: float = 0.05  # 5% vs 1.5% FTMO
    max_daily_loss: float = 0.15  # 15% vs 5% FTMO
    max_total_loss: float = 0.25  # 25% vs 10% FTMO
    max_portfolio_heat: float = 0.20  # 20% vs 6% FTMO

    # Execution parameters
    fees: float = 0.00005
    slippage: float = 0.0001

    # FTMO compliance (for comparison only)
    ftmo_daily_loss_limit: float = 0.05
    ftmo_total_loss_limit: float = 0.10
    ftmo_profit_target: float = 0.10


@dataclass
class StrategyDefaults:
    """🎯 FREQUENCY-OPTIMIZED Strategy defaults."""
    timeframe: str

    # 🚀 FILTER CONTROL - PERSONAL TRADING DEFAULTS
    use_htf_confirmation: bool = False  # 🔑 DISABLED - Key bottleneck removed
    stress_threshold: float = 4.0  # 🔑 RELAXED - vs 2.2 FTMO
    min_wick_ratio: float = 0.05  # 🔑 MINIMAL - vs 0.3 FTMO
    use_rejection_wicks: bool = False  # 🔑 DISABLED - no wick requirement
    use_session_filter: bool = False  # 🔑 DISABLED - 24/7 trading

    # RSI range - VERY WIDE
    rsi_min: int = 5  # vs 25 FTMO
    rsi_max: int = 95  # vs 75 FTMO

    # Volume filter - RELAXED
    volume_multiplier: float = 0.8  # vs 1.1 FTMO
    use_volume_filter: bool = False  # Optional volume filter

    # Bollinger Bands defaults - WIDER RANGES
    bb_window_range: List[int] = field(default_factory=lambda: [15, 20, 30, 50])
    bb_std_dev_range: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])

    # Risk management - PERSONAL TRADING
    sl_percent_range: List[float] = field(
        default_factory=lambda: [0.008, 0.01, 0.015, 0.02])
    tp_percent_range: List[float] = field(
        default_factory=lambda: [0.025, 0.03, 0.04, 0.05])

    # Order block defaults - RELAXED
    ob_lookback_range: List[int] = field(default_factory=lambda: [3, 5, 7, 10])
    ob_strength_range: List[float] = field(default_factory=lambda: [1.2, 1.5, 2.0])

    # LSTM defaults - RELAXED THRESHOLD
    lstm_threshold_range: List[float] = field(
        default_factory=lambda: [0.25, 0.35, 0.45])

    # Risk defaults - PERSONAL TRADING
    risk_per_trade_range: List[float] = field(
        default_factory=lambda: [0.03, 0.05, 0.07])

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
    """Centralized configuration manager - PERSONAL TRADING OPTIMIZED."""

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
                        'backtest': 'backtest_config.json',
                        'logging_config': 'logging_config.json'}

        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)
                        self._update_config(config_name, data)
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")
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
        """Initialize symbol configurations - MULTI-SYMBOL READY."""
        # 🚀 TARGET SYMBOLS FOR 250+ TRADES/YEAR
        default_symbols = ["GER40.cash", "XAUUSD", "EURUSD", "US30.cash", "GBPUSD"]

        # Try to get symbols from MT5
        mt5_symbols = self._fetch_mt5_symbols(default_symbols)

        # Create symbol configurations
        for symbol in mt5_symbols:
            symbol_info = self._get_mt5_symbol_info(symbol)
            if symbol_info:
                self.symbols[symbol] = SymbolConfig(symbol=symbol,
                                                    pip_value=symbol_info.get(
                                                        'pip_value', 10.0),
                                                    contract_size=symbol_info.get(
                                                        'contract_size', 1.0),
                                                    tick_size=symbol_info.get(
                                                        'tick_size', 0.1),
                                                    volume_min=symbol_info.get(
                                                        'volume_min', 0.01),
                                                    volume_max=symbol_info.get(
                                                        'volume_max', 100.0))
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
            print(f"Warning: Could not load symbols from MT5: {e}")

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
                        'tick_size': getattr(symbol_info, 'point', 0.0001),
                        'volume_min': symbol_info.volume_min,
                        'volume_max': symbol_info.volume_max}
                    mt5.shutdown()
                    return result
                mt5.shutdown()
        except Exception as e:
            print(f"Warning: Error getting symbol info for {symbol}: {e}")

        return None

    def _initialize_strategy_defaults(self) -> None:
        """🚀 FREQUENCY-OPTIMIZED strategy defaults by timeframe."""

        # M5 defaults - AGGRESSIVE FOR HIGH FREQUENCY
        self.strategy_defaults["M5"] = StrategyDefaults(timeframe="M5",
            use_htf_confirmation=False,  # 🔑 DISABLED
            stress_threshold=5.0,  # 🔑 VERY RELAXED
            min_wick_ratio=0.02,  # 🔑 MINIMAL
            use_rejection_wicks=False,  # 🔑 DISABLED
            use_session_filter=False,  # 🔑 24/7 TRADING
            rsi_min=3, rsi_max=97,  # 🔑 VERY WIDE
            volume_multiplier=0.7,  # 🔑 VERY RELAXED
            bb_window_range=[5, 10, 15, 20], sl_percent_range=[0.005, 0.008, 0.01],
            tp_percent_range=[0.01, 0.015, 0.02], risk_per_trade_range=[0.03, 0.05])

        # H1 defaults - PERSONAL TRADING OPTIMIZED
        self.strategy_defaults["H1"] = StrategyDefaults(timeframe="H1",
            use_htf_confirmation=False,  # 🔑 DISABLED - KEY FIX
            stress_threshold=4.0,  # 🔑 RELAXED vs 2.2
            min_wick_ratio=0.05,  # 🔑 MINIMAL vs 0.3
            use_rejection_wicks=False,  # 🔑 DISABLED
            use_session_filter=False,  # 🔑 24/7 TRADING
            rsi_min=5, rsi_max=95,  # 🔑 WIDE vs 25-75
            volume_multiplier=0.8,  # 🔑 RELAXED vs 1.1
            bb_window_range=[15, 20, 30, 50],
            sl_percent_range=[0.008, 0.01, 0.015, 0.02],
            tp_percent_range=[0.025, 0.03, 0.04, 0.05],
            risk_per_trade_range=[0.03, 0.05, 0.07]  # 🔑 PERSONAL RISK
        )

        # D1 defaults - RELAXED FOR SWING TRADING
        self.strategy_defaults["D1"] = StrategyDefaults(timeframe="D1",
            use_htf_confirmation=False,  # 🔑 DISABLED
            stress_threshold=3.5,  # 🔑 RELAXED
            min_wick_ratio=0.1,  # 🔑 RELAXED
            use_rejection_wicks=False,  # 🔑 DISABLED
            use_session_filter=False,  # 🔑 24/7 TRADING
            rsi_min=10, rsi_max=90,  # 🔑 WIDE
            volume_multiplier=0.9,  # 🔑 RELAXED
            bb_window_range=[20, 30, 40, 50], sl_percent_range=[0.015, 0.02, 0.03],
            tp_percent_range=[0.04, 0.05, 0.06],
            risk_per_trade_range=[0.03, 0.05, 0.07])

    # UNIFIED API METHODS
    def get_symbol_config(self, symbol: str) -> SymbolConfig:
        """Get symbol configuration."""
        return self.symbols.get(symbol, SymbolConfig(symbol=symbol))

    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        return self.get_symbol_config(symbol).pip_value

    def get_strategy_params(self, strategy_name: str, timeframe: str) -> Dict[str, Any]:
        """🚀 FREQUENCY-OPTIMIZED strategy parameters."""
        defaults = self.strategy_defaults.get(timeframe,
                                              self.strategy_defaults.get("H1"))
        symbol_config = self.get_symbol_config(self.trading.default_symbol)

        # 🚀 PERSONAL TRADING PARAMETERS
        params = {# Base trading config - PERSONAL ACCOUNT
            'symbol': self.trading.default_symbol,
            'initial_capital': self.trading.initial_capital,
            'risk_per_trade': self.trading.max_risk_per_trade,  # 5% vs 1.5%

            # 🔑 FREQUENCY CONTROL FILTERS - DISABLED/RELAXED
            'use_htf_confirmation': defaults.use_htf_confirmation,  # FALSE
            'stress_threshold': defaults.stress_threshold,  # 4.0 vs 2.2
            'min_wick_ratio': defaults.min_wick_ratio,  # 0.05 vs 0.3
            'use_rejection_wicks': defaults.use_rejection_wicks,  # FALSE
            'use_session_filter': defaults.use_session_filter,  # FALSE
            'rsi_min': defaults.rsi_min,  # 5 vs 25
            'rsi_max': defaults.rsi_max,  # 95 vs 75
            'volume_multiplier': defaults.volume_multiplier,  # 0.8 vs 1.1
            'use_volume_filter': defaults.use_volume_filter,  # FALSE

            # Symbol-specific
            'pip_value': symbol_config.pip_value,
            'contract_size': symbol_config.contract_size,

            # Strategy ranges
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
        """🚀 Get TARGET symbols for multi-symbol trading."""
        return list(self.symbols.keys())

    def save_config(self) -> None:
        """Save current configuration to files."""
        configs = {'trading': self.trading, 'backtest': self.backtest,
                   'logging_config': self.logging_config}

        for config_name, config_obj in configs.items():
            config_path = self.config_dir / f"{config_name}_config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(config_obj), f, indent=2)

        logger.info("🚀 Personal trading configuration saved")

    def reload_config(self) -> None:
        """Reload configuration from files."""
        self._load_configs()
        self._setup_logging()
        logger.info("Configuration reloaded")


# 🚀 PERSONAL TRADING CONFIGURATION INSTANCE
config_manager = UnifiedConfigManager()

# Create logger AFTER config_manager is initialized
logger = logging.getLogger(__name__)

# 🚀 PERSONAL TRADING EXPORTS
INITIAL_CAPITAL = config_manager.trading.initial_capital  # 100k vs 10k
FEES = config_manager.trading.fees
SYMBOLS = config_manager.get_available_symbols()  # Multi-symbol ready
OUTPUT_DIR = config_manager.get_output_dir()

# FTMO limits (for comparison)
MAX_DAILY_LOSS = config_manager.trading.ftmo_daily_loss_limit
MAX_TOTAL_LOSS = config_manager.trading.ftmo_total_loss_limit
PROFIT_TARGET = config_manager.trading.ftmo_profit_target

# 🚀 PERSONAL TRADING LIMITS
PERSONAL_MAX_RISK_PER_TRADE = config_manager.trading.max_risk_per_trade  # 5%
PERSONAL_MAX_DAILY_LOSS = config_manager.trading.max_daily_loss  # 15%
PERSONAL_MAX_TOTAL_LOSS = config_manager.trading.max_total_loss  # 25%


# UNIFIED API FUNCTIONS
def get_pip_value(symbol: str) -> float:
    """Get pip value for symbol."""
    return config_manager.get_pip_value(symbol)


def get_symbol_info(symbol: str) -> SymbolConfig:
    """Get complete symbol information."""
    return config_manager.get_symbol_config(symbol)


def get_strategy_config(strategy_name: str, timeframe: str) -> Dict[str, Any]:
    """🚀 Get FREQUENCY-OPTIMIZED configuration for strategy and timeframe."""
    return config_manager.get_strategy_params(strategy_name, timeframe)


def get_risk_config() -> Dict[str, float]:
    """🚀 Get PERSONAL TRADING risk management configuration."""
    return {'max_risk_per_trade': config_manager.trading.max_risk_per_trade,  # 5%
        'max_daily_loss': config_manager.trading.max_daily_loss,  # 15%
        'max_total_loss': config_manager.trading.max_total_loss,  # 25%
        'max_portfolio_heat': config_manager.trading.max_portfolio_heat  # 20%
    }


def get_multi_symbol_config() -> Dict[str, Any]:
    """🚀 Get multi-symbol trading configuration for 250+ trades/year."""
    return {'target_symbols': SYMBOLS,
        # ["GER40.cash", "XAUUSD", "EURUSD", "US30.cash", "GBPUSD"]
        'expected_trades_per_symbol': 60, 'total_expected_trades': 60 * len(SYMBOLS),
        # 300 trades/year
        'portfolio_risk': PERSONAL_MAX_RISK_PER_TRADE,  # 5% per trade
        'correlation_management': True}


# 🎯 LOG THE CHANGES
if __name__ == "__main__":
    logger.info("🚀 SOPHY4 CONFIGURATION UPDATED FOR PERSONAL TRADING")
    logger.info("=" * 60)
    logger.info(f"💰 Capital: ${INITIAL_CAPITAL:,.0f} (vs $10k FTMO)")
    logger.info(f"⚡ Risk/Trade: {PERSONAL_MAX_RISK_PER_TRADE:.1%} (vs 1.5% FTMO)")
    logger.info(f"🎯 Target Symbols: {len(SYMBOLS)} ({', '.join(SYMBOLS)})")

    multi_config = get_multi_symbol_config()
    logger.info(
        f"📈 Expected Frequency: {multi_config['total_expected_trades']} trades/year")
    logger.info("🔑 KEY FILTERS DISABLED:")
    logger.info("   ❌ HTF Confirmation = FALSE (was blocking 100%)")
    logger.info("   ⬆️ Stress Threshold = 4.0 (was 2.2)")
    logger.info("   ⬇️ Min Wick Ratio = 0.05 (was 0.3)")
    logger.info("   ❌ Session Filter = FALSE (24/7 trading)")
    logger.info("   📊 RSI Range = 5-95 (was 25-75)")
    logger.info("🚀 READY FOR 250+ TRADES/YEAR!")