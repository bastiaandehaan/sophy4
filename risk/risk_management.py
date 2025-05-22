# risk/risk_management.py - Enhanced Version
import logging
from typing import Dict, Optional, Tuple, Any, List
from enum import Enum
from dataclasses import dataclass
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from scipy.stats import norm
from config import logger


class RiskModel(Enum):
    """Risk calculation models."""
    FIXED_PERCENT = "fixed_percent"
    VAR_HISTORICAL = "var_historical"
    VAR_PARAMETRIC = "var_parametric"
    KELLY_CRITERION = "kelly_criterion"
    ADAPTIVE = "adaptive"


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_risk_per_trade: float = 0.02
    max_daily_loss: float = 0.05
    max_total_loss: float = 0.10
    max_portfolio_heat: float = 0.06  # Total risk across all positions
    max_correlation_exposure: float = 0.04  # Max risk in correlated assets


class EnhancedRiskManager:
    """
    Advanced risk management with multiple models and safety checks.
    """

    def __init__(self, confidence_level: float = 0.95,
            risk_model: RiskModel = RiskModel.ADAPTIVE,
            limits: Optional[RiskLimits] = None, lookback_periods: int = 252,
            correlation_threshold: float = 0.7):
        self.confidence_level = confidence_level
        self.risk_model = risk_model
        self.limits = limits or RiskLimits()
        self.lookback_periods = lookback_periods
        self.correlation_threshold = correlation_threshold

        # Caching
        self._var_cache: Dict[Tuple, float] = {}
        self._symbol_cache: Dict[str, Dict] = {}

        # Performance tracking
        self._trade_history: List[Dict] = []

        logger.info(f"RiskManager initialized with {risk_model.value} model")

    def calculate_position_size(self, symbol: str, capital: float, returns: pd.Series,
            entry_price: float, stop_loss_price: float,
            current_positions: Optional[Dict[str, float]] = None,
            strategy_performance: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size using selected risk model.

        Returns:
            Dict with position_size, risk_amount, confidence_score, warnings
        """
        result = {'position_size': 0.0, 'risk_amount': 0.0, 'confidence_score': 0.0,
            'warnings': [], 'model_used': self.risk_model.value}

        try:
            # Pre-flight safety checks
            safety_check = self._safety_checks(symbol, capital, current_positions)
            if not safety_check['passed']:
                result['warnings'] = safety_check['warnings']
                return result

            # Get symbol info
            symbol_info = self._get_symbol_info(symbol)
            if not symbol_info:
                result['warnings'].append(f"Cannot retrieve symbol info for {symbol}")
                return result

            # Calculate risk amount based on model
            if self.risk_model == RiskModel.FIXED_PERCENT:
                risk_amount = capital * self.limits.max_risk_per_trade

            elif self.risk_model == RiskModel.VAR_HISTORICAL:
                var_risk = self._calculate_var_historical(returns, capital)
                risk_amount = min(var_risk, capital * self.limits.max_risk_per_trade)

            elif self.risk_model == RiskModel.KELLY_CRITERION:
                kelly_risk = self._calculate_kelly_size(strategy_performance, capital)
                risk_amount = min(kelly_risk, capital * self.limits.max_risk_per_trade)

            elif self.risk_model == RiskModel.ADAPTIVE:
                risk_amount = self._calculate_adaptive_size(symbol, capital, returns,
                    strategy_performance, current_positions)
            else:
                risk_amount = capital * self.limits.max_risk_per_trade

            # Calculate position size from risk amount
            price_risk = abs(entry_price - stop_loss_price)
            if price_risk <= 0:
                result['warnings'].append("Invalid stop loss: no price risk")
                return result

            # Account for contract size and currency conversion
            contract_size = symbol_info.get('contract_size', 1.0)
            position_size = risk_amount / (price_risk * contract_size)

            # Apply position size limits
            min_size = symbol_info.get('volume_min', 0.01)
            max_size = symbol_info.get('volume_max', 100.0)
            position_size = max(min_size, min(position_size, max_size))

            # Portfolio heat check
            portfolio_heat = self._calculate_portfolio_heat(current_positions,
                                                            risk_amount)
            if portfolio_heat > self.limits.max_portfolio_heat:
                position_size *= (self.limits.max_portfolio_heat / portfolio_heat)
                result['warnings'].append(
                    f"Position reduced due to portfolio heat: {portfolio_heat:.2%}")

            # Correlation check
            correlation_risk = self._check_correlation_risk(symbol, current_positions)
            if correlation_risk > self.limits.max_correlation_exposure:
                position_size *= 0.5  # Reduce by half for correlated assets
                result['warnings'].append(f"Position reduced due to correlation risk")

            result.update(
                {'position_size': round(position_size, 2), 'risk_amount': risk_amount,
                    'confidence_score': self._calculate_confidence_score(returns,
                                                                         strategy_performance),
                    'portfolio_heat': portfolio_heat,
                    'correlation_risk': correlation_risk})

            logger.debug(
                f"Position size calculated for {symbol}: {position_size:.2f} lots")
            return result

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            result['warnings'].append(f"Calculation error: {str(e)}")
            return result

    def _calculate_adaptive_size(self, symbol: str, capital: float, returns: pd.Series,
            strategy_performance: Optional[Dict],
            current_positions: Optional[Dict]) -> float:
        """Adaptive risk sizing based on multiple factors."""
        base_risk = capital * self.limits.max_risk_per_trade

        # Factor 1: Volatility adjustment
        if len(returns) >= 20:
            current_vol = returns.tail(20).std()
            long_vol = returns.std()
            vol_ratio = current_vol / long_vol if long_vol > 0 else 1.0
            vol_adjustment = 1.0 / max(vol_ratio, 0.5)  # Reduce size in high vol
        else:
            vol_adjustment = 1.0

        # Factor 2: Strategy performance adjustment
        if strategy_performance:
            win_rate = strategy_performance.get('win_rate', 0.5)
            profit_factor = strategy_performance.get('profit_factor', 1.0)
            performance_score = (win_rate * profit_factor) / 0.5  # Normalized
            performance_adjustment = min(max(performance_score, 0.5), 1.5)
        else:
            performance_adjustment = 1.0

        # Factor 3: Market regime adjustment
        market_stress = self._detect_market_stress(returns)
        stress_adjustment = 0.5 if market_stress else 1.0

        # Combine all factors
        final_adjustment = vol_adjustment * performance_adjustment * stress_adjustment
        adjusted_risk = base_risk * final_adjustment

        logger.debug(f"Adaptive risk calculation: vol={vol_adjustment:.2f}, "
                     f"perf={performance_adjustment:.2f}, stress={stress_adjustment:.2f}")

        return min(adjusted_risk, capital * self.limits.max_risk_per_trade * 1.5)

    def _calculate_kelly_size(self, strategy_performance: Optional[Dict],
                              capital: float) -> float:
        """Calculate Kelly criterion position size."""
        if not strategy_performance:
            return capital * self.limits.max_risk_per_trade

        win_rate = strategy_performance.get('win_rate', 0.5)
        avg_win = strategy_performance.get('avg_winning_trade', 1.0)
        avg_loss = abs(strategy_performance.get('avg_losing_trade', -1.0))

        if avg_loss == 0:
            return capital * self.limits.max_risk_per_trade

        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Apply Kelly with safety factor (quarter Kelly)
        safe_kelly = max(kelly_fraction * 0.25, 0.005)
        return min(safe_kelly * capital, capital * self.limits.max_risk_per_trade * 2)

    def _safety_checks(self, symbol: str, capital: float,
                       current_positions: Optional[Dict]) -> Dict:
        """Pre-flight safety checks."""
        warnings = []

        # Capital check
        if capital <= 0:
            warnings.append("Invalid capital amount")

        # Market hours check
        if not self._is_market_open(symbol):
            warnings.append(f"Market closed for {symbol}")

        # Daily loss check
        daily_pnl = self._get_daily_pnl(current_positions)
        if daily_pnl < -capital * self.limits.max_daily_loss:
            warnings.append("Daily loss limit exceeded")

        return {'passed': len(warnings) == 0, 'warnings': warnings}

    def _detect_market_stress(self, returns: pd.Series) -> bool:
        """Detect high-stress market conditions."""
        if len(returns) < 20:
            return False

        # Check for extreme volatility spikes
        recent_vol = returns.tail(5).std()
        normal_vol = returns.tail(60).std()

        return recent_vol > normal_vol * 2.0

    def _calculate_portfolio_heat(self, current_positions: Optional[Dict],
                                  new_risk: float) -> float:
        """Calculate total portfolio risk exposure."""
        if not current_positions:
            return new_risk / 10000  # Assume 10k capital for percentage

        total_risk = sum(
            pos.get('risk_amount', 0) for pos in current_positions.values())
        return (total_risk + new_risk) / 10000

    def _check_correlation_risk(self, symbol: str,
                                current_positions: Optional[Dict]) -> float:
        """Check correlation risk with existing positions."""
        # Simplified correlation mapping
        correlations = {'EURUSD': ['GBPUSD', 'EURGBP', 'USDCHF'],
            'GER40.cash': ['FRA40.cash', 'UK100.cash'], 'XAUUSD': ['XAGUSD']}

        if not current_positions:
            return 0.0

        correlated_symbols = correlations.get(symbol, [])
        correlated_risk = sum(
            pos.get('risk_amount', 0) for sym, pos in current_positions.items() if
            sym in correlated_symbols)

        return correlated_risk / 10000  # Convert to percentage

    def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get and cache symbol information."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        try:
            if not mt5.initialize():
                return None

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None

            info = {'contract_size': symbol_info.trade_contract_size,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'tick_size': symbol_info.tick_size,
                'tick_value': getattr(symbol_info, 'trade_tick_value', 10.0)}

            self._symbol_cache[symbol] = info
            return info

        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
        finally:
            mt5.shutdown()

    def _is_market_open(self, symbol: str) -> bool:
        """Check if market is open for symbol."""
        try:
            if not mt5.initialize():
                return False

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False

            # Basic check: if we can get recent tick data
            return True

        except Exception:
            return False
        finally:
            mt5.shutdown()

    def _get_daily_pnl(self, current_positions: Optional[Dict]) -> float:
        """Get current daily P&L."""
        if not current_positions:
            return 0.0

        return sum(pos.get('unrealized_pnl', 0) for pos in current_positions.values())

    def _calculate_confidence_score(self, returns: pd.Series,
                                    strategy_performance: Optional[Dict]) -> float:
        """Calculate confidence score for the trade setup."""
        score = 0.5  # Base score

        # Data quality factor
        if len(returns) >= 100:
            score += 0.2

        # Strategy performance factor
        if strategy_performance:
            win_rate = strategy_performance.get('win_rate', 0.5)
            if win_rate > 0.6:
                score += 0.2
            elif win_rate < 0.4:
                score -= 0.2

        # Volatility factor
        if len(returns) >= 20:
            vol_ratio = returns.tail(10).std() / returns.std()
            if vol_ratio < 1.2:  # Low current volatility
                score += 0.1

        return max(0.0, min(1.0, score))

    def update_trade_result(self, symbol: str, entry_price: float, exit_price: float,
                            position_size: float, result: str) -> None:
        """Update trade history for performance tracking."""
        trade = {'symbol': symbol, 'entry_price': entry_price, 'exit_price': exit_price,
            'position_size': position_size,
            'pnl': (exit_price - entry_price) * position_size, 'result': result,
            # 'win' or 'loss'
            'timestamp': pd.Timestamp.now()}

        self._trade_history.append(trade)

        # Keep only recent trades for performance calculation
        if len(self._trade_history) > 100:
            self._trade_history = self._trade_history[-100:]

    def get_performance_stats(self) -> Dict[str, float]:
        """Get recent performance statistics."""
        if not self._trade_history:
            return {}

        df = pd.DataFrame(self._trade_history)

        return {'win_rate': len(df[df['result'] == 'win']) / len(df),
            'avg_winning_trade': df[df['pnl'] > 0]['pnl'].mean(),
            'avg_losing_trade': df[df['pnl'] < 0]['pnl'].mean(), 'profit_factor': abs(
                df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if len(
                df[df['pnl'] < 0]) > 0 else 0, 'total_trades': len(df)}