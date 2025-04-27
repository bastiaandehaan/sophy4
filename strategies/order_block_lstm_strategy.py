# Aangepaste versie van generate_signals methode voor OrderBlockLSTMStrategy
# Te implementeren in strategies/order_block_lstm_strategy.py

def generate_signals(self, df: pd.DataFrame, current_capital: float = None) -> \
        Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Generate entry, stop-loss, and take-profit signals with improved criteria for more signals.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data.
        current_capital (float, optional): Current account capital. Defaults to initial_capital.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (entries, sl_stop, tp_stop).
        entries: 1 for buy, -1 for sell, 0 for hold.
    """
    print(
        f"DEBUG: Analysing {len(df)} candles for OrderBlockLSTMStrategy with {self.symbol}")

    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"DataFrame missing required columns: {required_columns} for {self.symbol}")

    entries = pd.Series(0, index=df.index, dtype=int)
    sl_stop = pd.Series(self.sl_fixed_percent, index=df.index)
    tp_stop = pd.Series(self.tp_fixed_percent, index=df.index)

    if self.verbose_logging:
        print(f"\nGenerating signals for {len(df)} candles ({self.symbol})...")
    logger.info(f"Generating signals for {len(df)} candles ({self.symbol})")

    capital = current_capital or self.initial_capital

    max_value = capital
    if self.risk_manager.monitor_drawdown(capital, max_value):
        logger.warning(
            f"Drawdown limit exceeded for {self.symbol}, no new signals generated")
        return entries, sl_stop, tp_stop

    max_daily_loss = self.risk_manager.get_max_daily_loss(capital)
    max_total_loss = self.risk_manager.get_max_total_loss(capital)
    logger.info(
        f"FTMO limits for {self.symbol}: Max daily loss={max_daily_loss:.2f}, Max total loss={max_total_loss:.2f}")

    order_blocks = self.detect_order_blocks(df)
    print(f"DEBUG: Detected {len(order_blocks)} order blocks")

    if not order_blocks:
        if self.verbose_logging:
            print(
                f"No order blocks detected for {self.symbol}, returning empty signals")
        logger.info(
            f"No order blocks detected for {self.symbol}, returning empty signals")
        return entries, sl_stop, tp_stop

    lstm_pred = 0.0
    if self.model is not None:
        try:
            X_in = self.prepare_lstm_input(df)
            lstm_pred = self.model.predict(X_in, verbose=0)[0][0]
            if self.verbose_logging:
                print(
                    f"LSTM prediction: {lstm_pred:.4f}, threshold: {self.lstm_threshold} ({self.symbol})")
            logger.info(
                f"LSTM prediction: {lstm_pred:.4f}, threshold: {self.lstm_threshold} ({self.symbol})")
        except Exception as e:
            logger.error(f"LSTM prediction failed for {self.symbol}: {str(e)}")
            if self.verbose_logging:
                print(f"LSTM prediction failed for {self.symbol}: {str(e)}")
    else:
        if self.verbose_logging:
            print(
                f"No LSTM model available for {self.symbol}, using default LSTM value of 0")
        logger.warning(
            f"No LSTM model available for {self.symbol}, using default LSTM value of 0")

    # Get current price for trading conditions
    current_price = df['close'].iloc[-1]

    # Prepare returns for risk calculation
    returns = df['close'].pct_change().dropna()
    open_positions = {}

    # Track signaling metrics
    signals_checked = 0
    recent_ob_count = 0
    signals_generated = 0

    # VERBETERD: Vergroot de historische order block periode - gebruik alles vanaf het begin
    time_filter = df.index[0]

    # Loop through detected order blocks
    for ob in order_blocks:
        signals_checked += 1

        # Skip very old order blocks
        if ob.time < time_filter:
            continue

        recent_ob_count += 1

        # Get past data for Fibonacci levels
        past_data = df.loc[:ob.time].tail(self.fib_lookback)
        if past_data.empty:
            continue

        swing_high = past_data['high'].max()
        swing_low = past_data['low'].min()
        fib = self.calculate_fibonacci_levels(swing_high, swing_low)

        if self.verbose_logging and (
                signals_checked <= 10 or signals_checked % 50 == 0):
            print(
                f"\nOrder Block #{signals_checked}: {'Bullish' if ob.direction == 1 else 'Bearish'} at {ob.time} ({self.symbol})")
            print(f"  Price range: {ob.low:.5f}-{ob.high:.5f}")
            print(
                f"  Fibonacci levels: 23.6%={fib['23.6%']:.5f}, 38.2%={fib['38.2%']:.5f}, 50.0%={fib['50.0%']:.5f}, 61.8%={fib['61.8%']:.5f}")
            print(f"  Current price: {current_price:.5f}")

        # VERBETERD: Meer versoepelde criteria voor prijsafstanden en Fibonacci zones
        # voor meer handelssignalen
        if ob.direction == 1:  # Bullish
            # Verruimd van 5% naar 10%
            price_near_ob = abs(current_price - ob.low) / ob.low < 0.10
            # Bredere Fibonacci zone
            fib_zone = (fib['61.8%'] * 0.7) <= current_price <= (fib['38.2%'] * 1.3)
            # VERBETERD: LSTM voorspelling gebruiken indien beschikbaar
            lstm_trend_ok = lstm_pred > self.lstm_threshold if self.model is not None else True
            trading_condition = price_near_ob and (fib_zone or lstm_trend_ok) and not ob.traded

            if self.verbose_logging and (
                    signals_checked <= 5 or signals_checked % 50 == 0):
                print(
                    f"  Bullish conditions: price_near_ob={price_near_ob}, fib_zone={fib_zone}, lstm_trend_ok={lstm_trend_ok}")

        elif ob.direction == -1:  # Bearish
            # Verruimd van 5% naar 10%
            price_near_ob = abs(current_price - ob.high) / ob.high < 0.10
            # Bredere Fibonacci zone
            fib_zone = (fib['38.2%'] * 0.7) <= current_price <= (fib['61.8%'] * 1.3)
            # VERBETERD: LSTM voorspelling gebruiken indien beschikbaar
            lstm_trend_ok = lstm_pred > self.lstm_threshold if self.model is not None else True
            trading_condition = price_near_ob and (fib_zone or lstm_trend_ok) and not ob.traded

            if self.verbose_logging and (
                    signals_checked <= 5 or signals_checked % 50 == 0):
                print(
                    f"  Bearish conditions: price_near_ob={price_near_ob}, fib_zone={fib_zone}, lstm_trend_ok={lstm_trend_ok}")

        # Signal placement (unchanged)
        if trading_condition:
            try:
                signal_idx = df.index.get_indexer([ob.time], method='pad')[0]
                if signal_idx < len(entries) - 1:
                    entries.iloc[signal_idx + 1] = ob.direction
                    signals_generated += 1
                    ob.traded = True

                    if self.verbose_logging:
                        print(
                            f"✅ {'Bullish' if ob.direction == 1 else 'Bearish'} signal at {df.index[signal_idx + 1]}, index={signal_idx + 1}")
                    logger.info(
                        f"Signal placed at index {signal_idx + 1} from OB at {ob.time}")
            except Exception as e:
                logger.error(f"Error placing signal: {str(e)}")

    # Summary statistics
    if self.verbose_logging:
        print(f"\nSignal generation results for {self.symbol}:")
        print(f"  Order Blocks analyzed: {signals_checked}")
        print(f"  Recent Order Blocks: {recent_ob_count}")
        print(f"  Signals generated: {signals_generated}")

    logger.info(
        f"Signal generation results for {self.symbol}: {signals_generated} signals from {recent_ob_count} recent OBs")
    self.risk_manager.clear_cache()
    return entries, sl_stop, tp_stop