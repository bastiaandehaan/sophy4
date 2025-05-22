# utils/logging_setup.py - Enhanced Logging System
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ContextualLogger:
    """Logger with trading-specific context."""

    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}

    def _format_message(self, message: str,
                        extra_context: Optional[Dict] = None) -> str:
        """Format message with context."""
        full_context = {**self.context, **(extra_context or {})}
        if full_context:
            context_str = " | ".join([f"{k}={v}" for k, v in full_context.items()])
            return f"{message} | {context_str}"
        return message

    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message, kwargs))

    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message, kwargs))

    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message(message, kwargs))

    def error(self, message: str, **kwargs):
        self.logger.error(self._format_message(message, kwargs))

    def critical(self, message: str, **kwargs):
        self.logger.critical(self._format_message(message, kwargs))

    def log_trade(self, action: str, symbol: str, price: float, size: float, **kwargs):
        """Log trading-specific events."""
        trade_context = {'action': action, 'symbol': symbol, 'price': price,
            'size': size, 'timestamp': datetime.now().isoformat(), **kwargs}
        self.info(f"TRADE_{action.upper()}", **trade_context)

    def log_signal(self, strategy: str, symbol: str, signal_type: str,
                   confidence: float, **kwargs):
        """Log trading signals."""
        signal_context = {'strategy': strategy, 'symbol': symbol,
            'signal_type': signal_type, 'confidence': confidence,
            'timestamp': datetime.now().isoformat(), **kwargs}
        self.info(f"SIGNAL_{signal_type.upper()}", **signal_context)

    def log_performance(self, strategy: str, symbol: str, metrics: Dict[str, float],
                        **kwargs):
        """Log performance metrics."""
        perf_context = {'strategy': strategy, 'symbol': symbol,
            'timestamp': datetime.now().isoformat(), **metrics, **kwargs}
        self.info("PERFORMANCE_UPDATE", **perf_context)


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname, 'logger': record.name,
            'message': record.getMessage(), 'module': record.module,
            'function': record.funcName, 'line': record.lineno}

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                           'filename', 'module', 'lineno', 'funcName', 'created',
                           'msecs', 'relativeCreated', 'thread', 'threadName',
                           'processName', 'process', 'message', 'exc_info', 'exc_text',
                           'stack_info']:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class TradingLogFilter(logging.Filter):
    """Filter for trading-specific logs."""

    def __init__(self, include_patterns: list = None, exclude_patterns: list = None):
        super().__init__()
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()

        # Exclude patterns take precedence
        for pattern in self.exclude_patterns:
            if pattern.lower() in message.lower():
                return False

        # If include patterns are specified, message must match one
        if self.include_patterns:
            return any(
                pattern.lower() in message.lower() for pattern in self.include_patterns)

        return True


def setup_enhanced_logging(log_dir: Path = None, log_level: LogLevel = LogLevel.INFO,
        enable_console: bool = True, enable_file: bool = True,
        enable_structured: bool = False, max_file_size_mb: int = 10,
        backup_count: int = 5) -> None:
    """
    Setup enhanced logging system.

    Args:
        log_dir: Directory for log files
        log_level: Minimum log level
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_structured: Use JSON structured logging
        max_file_size_mb: Maximum size per log file
        backup_count: Number of backup files to keep
    """
    if log_dir is None:
        log_dir = Path("logs")

    log_dir.mkdir(exist_ok=True, parents=True)

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root level
    root_logger.setLevel(getattr(logging, log_level.value))

    # Formatters
    if enable_structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s')

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.value))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handlers
    if enable_file:
        # Main log file with rotation
        main_handler = logging.handlers.RotatingFileHandler(log_dir / "sophy4.log",
            maxBytes=max_file_size_mb * 1024 * 1024, backupCount=backup_count)
        main_handler.setLevel(getattr(logging, log_level.value))
        main_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(main_handler)

        # Error-only log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "sophy4_errors.log", maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)

        # Trading-specific log file
        trading_handler = logging.handlers.RotatingFileHandler(
            log_dir / "sophy4_trading.log", maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count)
        trading_handler.setLevel(logging.INFO)
        trading_handler.setFormatter(detailed_formatter)

        # Add filter for trading-related logs
        trading_filter = TradingLogFilter(
            include_patterns=['TRADE_', 'SIGNAL_', 'PERFORMANCE_', 'BACKTEST',
                              'STRATEGY'])
        trading_handler.addFilter(trading_filter)
        root_logger.addHandler(trading_handler)

    # Performance log file (separate for analysis)
    if enable_file:
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir / "sophy4_performance.log", maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count)
        perf_handler.setLevel(logging.INFO)

        if enable_structured:
            perf_handler.setFormatter(StructuredFormatter())
        else:
            perf_handler.setFormatter(logging.Formatter('%(message)s'))

        perf_filter = TradingLogFilter(include_patterns=['PERFORMANCE_', 'TRADE_'])
        perf_handler.addFilter(perf_filter)
        root_logger.addHandler(perf_handler)


def get_strategy_logger(strategy_name: str, symbol: str = "",
                        timeframe: str = "") -> ContextualLogger:
    """Get logger with strategy context."""
    context = {'strategy': strategy_name}
    if symbol:
        context['symbol'] = symbol
    if timeframe:
        context['timeframe'] = timeframe

    return ContextualLogger(f"strategy.{strategy_name}", context)


def get_backtest_logger(strategy_name: str, symbol: str,
                        timeframe: str) -> ContextualLogger:
    """Get logger with backtest context."""
    context = {'strategy': strategy_name, 'symbol': symbol, 'timeframe': timeframe,
        'operation': 'backtest'}

    return ContextualLogger("backtest", context)


def get_optimization_logger(strategy_name: str) -> ContextualLogger:
    """Get logger with optimization context."""
    context = {'strategy': strategy_name, 'operation': 'optimization'}

    return ContextualLogger("optimization", context)


# Performance monitoring decorator
def log_performance(logger: ContextualLogger = None):
    """Decorator to log function performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            import time

            start_time = time.time()

            if logger:
                logger.debug(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                if logger:
                    logger.info(f"Completed {func.__name__}",
                                execution_time=f"{execution_time:.3f}s")

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                if logger:
                    logger.error(f"Failed {func.__name__}",
                                 execution_time=f"{execution_time:.3f}s", error=str(e))
                raise

        return wrapper

    return decorator


# Usage examples:
"""
# Setup logging
setup_enhanced_logging(
    log_dir=Path("logs"),
    log_level=LogLevel.INFO,
    enable_structured=True
)

# Get contextual loggers
strategy_logger = get_strategy_logger("BollongStrategy", "EURUSD", "H1")
backtest_logger = get_backtest_logger("BollongStrategy", "EURUSD", "H1")

# Log trading events
strategy_logger.log_signal("BollongStrategy", "EURUSD", "LONG", 0.75, 
                          bollinger_position=0.8, rsi=45)

strategy_logger.log_trade("ENTRY", "EURUSD", 1.1234, 0.1, 
                         stop_loss=1.1200, take_profit=1.1300)

# Log performance
backtest_logger.log_performance("BollongStrategy", "EURUSD", {
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.05,
    "win_rate": 0.6
})

# Use performance decorator
@log_performance(strategy_logger)
def complex_calculation():
    # Long-running calculation
    pass
"""