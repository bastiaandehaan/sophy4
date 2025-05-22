# utils/error_handling.py - Robust Error Handling
import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorResult:
    """Standardized error result."""
    success: bool
    data: Any = None
    error_message: str = ""
    error_code: str = ""
    severity: ErrorSeverity = ErrorSeverity.LOW
    retry_possible: bool = False
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


class TradingError(Exception):
    """Base exception for trading operations."""

    def __init__(self, message: str, error_code: str = "",
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity


class DataError(TradingError):
    """Data-related errors."""
    pass


class StrategyError(TradingError):
    """Strategy-related errors."""
    pass


class RiskError(TradingError):
    """Risk management errors."""
    pass


class BacktestError(TradingError):
    """Backtesting errors."""
    pass


def safe_execute(fallback_value: Any = None, reraise_on: Tuple[Exception, ...] = (),
        log_errors: bool = True,
        error_severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Callable:
    """
    Decorator for safe function execution with standardized error handling.

    Args:
        fallback_value: Value to return on error
        reraise_on: Exception types to re-raise instead of catching
        log_errors: Whether to log errors
        error_severity: Default severity level for errors
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> ErrorResult:
            try:
                result = func(*args, **kwargs)
                return ErrorResult(success=True, data=result)

            except reraise_on:
                raise

            except Exception as e:
                error_message = str(e)
                error_code = f"{func.__name__}_error"

                if log_errors:
                    logger.error(f"Error in {func.__name__}: {error_message}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")

                # Determine if retry is possible
                retry_possible = not isinstance(e,
                                                (ValueError, TypeError, AttributeError))

                return ErrorResult(success=False, data=fallback_value,
                    error_message=error_message, error_code=error_code,
                    severity=error_severity, retry_possible=retry_possible,
                    context={'function': func.__name__, 'args': str(args)[:100],
                        # Truncate for logging
                        'kwargs': str(kwargs)[:100]})

        return wrapper

    return decorator


def validate_data(data: Any, validations: Dict[str, Callable]) -> ErrorResult:
    """
    Validate data against a set of validation functions.

    Args:
        data: Data to validate
        validations: Dict of validation_name -> validation_function

    Returns:
        ErrorResult with validation status
    """
    if data is None:
        return ErrorResult(success=False, error_message="Data is None",
            error_code="data_none", severity=ErrorSeverity.HIGH)

    failed_validations = []

    for validation_name, validation_func in validations.items():
        try:
            if not validation_func(data):
                failed_validations.append(validation_name)
        except Exception as e:
            logger.warning(f"Validation '{validation_name}' failed with error: {e}")
            failed_validations.append(f"{validation_name}_error")

    if failed_validations:
        return ErrorResult(success=False,
            error_message=f"Validation failed: {', '.join(failed_validations)}",
            error_code="validation_failed", severity=ErrorSeverity.MEDIUM,
            context={'failed_validations': failed_validations})

    return ErrorResult(success=True, data=data)


# Common validation functions
def validate_dataframe(df) -> bool:
    """Validate DataFrame is not empty and has required columns."""
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return False
    required_columns = ['open', 'high', 'low', 'close']
    return all(col in df.columns for col in required_columns)


def validate_positive_number(value) -> bool:
    """Validate value is a positive number."""
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False


def validate_percentage(value) -> bool:
    """Validate value is between 0 and 1."""
    try:
        return 0 <= float(value) <= 1
    except (ValueError, TypeError):
        return False


def validate_symbol(symbol) -> bool:
    """Validate trading symbol format."""
    if not isinstance(symbol, str):
        return False
    return len(symbol) >= 3 and symbol.replace('.', '').replace('_', '').isalnum()


# Enhanced backtest function with error handling
@safe_execute(fallback_value=(None, {}), error_severity=ErrorSeverity.HIGH)
def run_safe_backtest(strategy_name: str, parameters: Dict[str, Any], symbol: str,
                      **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Run backtest with comprehensive error handling.
    """
    # Input validation
    validations = {'valid_symbol': lambda x: validate_symbol(symbol),
        'valid_parameters': lambda x: isinstance(parameters, dict),
        'valid_strategy': lambda x: isinstance(strategy_name, str) and len(
            strategy_name) > 0}

    validation_result = validate_data((strategy_name, parameters, symbol),
                                      lambda x: all(
                                          validations[k](x) for k in validations))

    if not validation_result.success:
        raise DataError(f"Input validation failed: {validation_result.error_message}")

    # Import here to avoid circular imports
    from backtest.backtest import run_extended_backtest

    # Run the actual backtest
    result = run_extended_backtest(strategy_name=strategy_name, parameters=parameters,
        symbol=symbol, **kwargs)

    if result is None or len(result) != 2:
        raise BacktestError("Backtest returned invalid result")

    pf, metrics = result

    if pf is None:
        raise BacktestError(
            f"Failed to create portfolio for {strategy_name} on {symbol}")

    if not metrics:
        raise BacktestError("No metrics generated from backtest")

    # Validate metrics
    required_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
    missing_metrics = [m for m in required_metrics if m not in metrics]
    if missing_metrics:
        logger.warning(f"Missing metrics: {missing_metrics}")

    return pf, metrics


# Retry decorator for transient failures
def retry_on_failure(max_retries: int = 3, delay: float = 1.0,
                     exponential_backoff: bool = True) -> Callable:
    """
    Retry function on failure with configurable backoff.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries")
                        raise

                    wait_time = delay * (2 ** attempt if exponential_backoff else 1)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, "
                                   f"retrying in {wait_time}s: {str(e)}")

                    import time
                    time.sleep(wait_time)

            raise last_exception

        return wrapper

    return decorator


# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in code blocks."""

    def __init__(self, operation_name: str, reraise: bool = False,
                 log_errors: bool = True):
        self.operation_name = operation_name
        self.reraise = reraise
        self.log_errors = log_errors
        self.error_result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_message = str(exc_val)

            if self.log_errors:
                logger.error(f"Error in {self.operation_name}: {error_message}")
                logger.debug(f"Traceback: {traceback.format_exc()}")

            self.error_result = ErrorResult(success=False, error_message=error_message,
                error_code=f"{self.operation_name}_error",
                severity=ErrorSeverity.MEDIUM,
                context={'operation': self.operation_name})

            if not self.reraise:
                return True  # Suppress exception

        return False


# Usage examples:
"""
# Using safe_execute decorator
@safe_execute(fallback_value=pd.DataFrame(), error_severity=ErrorSeverity.HIGH)
def fetch_data(symbol: str) -> pd.DataFrame:
    # Data fetching logic that might fail
    pass

# Using error context
with ErrorContext("data_processing") as ctx:
    # Code that might fail
    result = process_data(df)

if ctx.error_result:
    logger.error(f"Processing failed: {ctx.error_result.error_message}")

# Using validation
result = validate_data(my_dataframe, {
    'not_empty': validate_dataframe,
    'valid_symbol': lambda df: validate_symbol(df.attrs.get('symbol', ''))
})

if not result.success:
    handle_error(result)
"""