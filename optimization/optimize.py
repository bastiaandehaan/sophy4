def walk_forward_test(strategy_name, symbol, params, period_days=1095):
    """
    Perform walk-forward testing to validate strategy robustness.

    Walk-forward testing divides historical data into multiple in-sample/out-of-sample
    segments and evaluates if strategy parameters optimized on in-sample data
    perform well on out-of-sample data.

    Args:
        strategy_name (str): Name of the strategy to test
        symbol (str): Trading symbol to test on
        params (dict): Strategy parameters to test
        period_days (int): Total period for testing in days

    Returns:
        dict: Results of walk-forward testing including validation status and metrics
    """
    import datetime as dt
    from backtest.extended_backtest import run_extended_backtest
    from config import logger

    logger.info(f"Running walk-forward test for {strategy_name} on {symbol}")

    # Get current date and calculate start date
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=period_days)

    # Number of windows to test (e.g., 3 windows = 3 in-sample/out-of-sample pairs)
    n_windows = 3
    window_size = period_days // n_windows

    in_sample_results = []
    out_of_sample_results = []

    # For each window
    for i in range(n_windows):
        # Calculate window dates
        window_start = start_date + dt.timedelta(days=i * window_size)
        in_sample_end = window_start + dt.timedelta(
            days=window_size * 0.7)  # 70% for in-sample
        out_sample_end = window_start + dt.timedelta(days=window_size)

        logger.info(
            f"Window {i + 1}: In-sample {window_start.date()} to {in_sample_end.date()}, "
            f"Out-of-sample {in_sample_end.date()} to {out_sample_end.date()}")

        # Run in-sample backtest
        in_sample_pf, in_sample_metrics = run_extended_backtest(
            strategy_name=strategy_name, parameters=params, symbol=symbol,
            start_date=window_start, end_date=in_sample_end)

        # Run out-of-sample backtest
        out_sample_pf, out_sample_metrics = run_extended_backtest(
            strategy_name=strategy_name, parameters=params, symbol=symbol,
            start_date=in_sample_end, end_date=out_sample_end)

        # Store results
        if in_sample_metrics and out_sample_metrics:
            in_sample_results.append(in_sample_metrics)
            out_of_sample_results.append(out_sample_metrics)

    # Analyze results
    if len(out_of_sample_results) < 1:
        logger.warning("Not enough data for walk-forward testing")
        return {"walk_forward_validated": False,
                "reason": "Insufficient data for testing"}

    # Calculate average metrics
    avg_in_sample_sharpe = sum(
        r.get('sharpe_ratio', 0) for r in in_sample_results) / len(in_sample_results)
    avg_out_sample_sharpe = sum(
        r.get('sharpe_ratio', 0) for r in out_of_sample_results) / len(
        out_of_sample_results)

    avg_in_sample_profit_factor = sum(
        r.get('profit_factor', 0) for r in in_sample_results) / len(in_sample_results)
    avg_out_sample_profit_factor = sum(
        r.get('profit_factor', 0) for r in out_of_sample_results) / len(
        out_of_sample_results)

    avg_in_sample_win_rate = sum(r.get('win_rate', 0) for r in in_sample_results) / len(
        in_sample_results)
    avg_out_sample_win_rate = sum(
        r.get('win_rate', 0) for r in out_of_sample_results) / len(
        out_of_sample_results)

    # Check if strategy is robust (out-of-sample performance is at least 70% of in-sample)
    sharpe_ratio_check = avg_out_sample_sharpe >= (avg_in_sample_sharpe * 0.7)
    profit_factor_check = avg_out_sample_profit_factor >= (
            avg_in_sample_profit_factor * 0.7)
    win_rate_check = avg_out_sample_win_rate >= (avg_in_sample_win_rate * 0.7)

    is_validated = sharpe_ratio_check and profit_factor_check and win_rate_check

    return {"walk_forward_validated": is_validated,
            "in_sample_performance": {"sharpe_ratio": avg_in_sample_sharpe,
                                      "profit_factor": avg_in_sample_profit_factor,
                                      "win_rate": avg_in_sample_win_rate},
            "out_of_sample_performance": {"sharpe_ratio": avg_out_sample_sharpe,
                                          "profit_factor": avg_out_sample_profit_factor,
                                          "win_rate": avg_out_sample_win_rate},
            "ratio_checks": {"sharpe_ratio_check": sharpe_ratio_check,
                             "profit_factor_check": profit_factor_check,
                             "win_rate_check": win_rate_check}}


def multi_instrument_test(strategy_name, params, symbols=None, timeframes=None):
    """
    Test a strategy with the same parameters across multiple instruments to check robustness.

    Args:
        strategy_name (str): Name of the strategy to test
        params (dict): Strategy parameters to test
        symbols (list): List of symbols to test on; defaults to a predefined list
        timeframes (dict): Dictionary mapping timeframe names to MT5 timeframe values

    Returns:
        dict: Results across all symbols and timeframes with performance metrics
    """
    from config import logger, SYMBOL
    from backtest.extended_backtest import run_extended_backtest
    import MetaTrader5 as mt5

    if symbols is None:
        symbols = [SYMBOL, 'US30.cash', 'EURUSD', 'GBPUSD']

    if timeframes is None:
        timeframes = {'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                      'D1': mt5.TIMEFRAME_D1}

    results = {}

    logger.info(f"Running multi-instrument test for {strategy_name}")
    logger.info(f"Parameters: {params}")
    logger.info(f"Testing on symbols: {symbols}")
    logger.info(f"Testing on timeframes: {list(timeframes.keys())}")

    # Track overall statistics
    total_tests = 0
    profitable_tests = 0

    # Test each symbol and timeframe combination
    for symbol in symbols:
        symbol_results = {}

        for tf_name, tf_value in timeframes.items():
            logger.info(f"Testing {strategy_name} on {symbol} ({tf_name})")

            # Run backtest with provided parameters
            pf, metrics = run_extended_backtest(strategy_name=strategy_name,
                                                parameters=params, symbol=symbol, timeframe=tf_value)

            # Store results and update stats
            if metrics:
                symbol_results[tf_name] = metrics
                total_tests += 1

                if metrics.get('net_profit', 0) > 0:
                    profitable_tests += 1

        # Store results for this symbol
        results[symbol] = symbol_results

    # Calculate robustness score
    robustness_score = (profitable_tests / total_tests) if total_tests > 0 else 0

    # Average performance metrics across all tests
    all_metrics = [metric for symbol_data in results.values() for metric in
                   symbol_data.values() if metric]

    avg_metrics = {}

    if all_metrics:
        for key in ['sharpe_ratio', 'profit_factor', 'win_rate', 'max_drawdown_pct']:
            values = [m.get(key, 0) for m in all_metrics]
            avg_metrics[key] = sum(values) / len(values) if values else 0

    return {"results_by_symbol": results, "robustness_score": robustness_score,
            "profitable_percentage": (
                    profitable_tests / total_tests * 100) if total_tests > 0 else 0,
            "average_metrics": avg_metrics, "total_tests": total_tests,
            "profitable_tests": profitable_tests}