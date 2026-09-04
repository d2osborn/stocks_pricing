import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import ftplib
import io

def calculate_adx(df, period=14):
    df = df.copy()

    high, low, close = df['High'], df['Low'], df['Close']

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    # Directional movement (compare the ORIGINAL moves, not mutated ones)
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder's smoothing (RMA) = ewm with alpha = 1/period, adjust=False
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    df['+DI'] = 100 * (plus_dm_s / atr)
    df['-DI'] = 100 * (minus_dm_s / atr)

    dx = 100 * (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'])
    df['DX'] = dx
    df['ADX'] = dx.ewm(alpha=alpha, adjust=False).mean()   # ADX is Wilder-smoothed too

    return df

# ==========================================
# WEEKLY TREND HELPERS
# ==========================================
def calculate_weekly_trend(df, fast_period=10, slow_period=30):
    """
    Resamples daily OHLCV bars into weekly bars (weeks ending Friday) and returns
    the weekly fast SMA and weekly slow EMA as two Series indexed by week-ending date.

    The final week is included even while it is still forming, which mirrors what
    you would see on a live weekly chart. Returns (None, None) when there is not
    enough weekly history to compute the requested periods.
    """
    if df is None or df.empty:
        return None, None

    needed = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    if 'Close' not in needed:
        return None, None

    try:
        weekly = df[needed].resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna(subset=['Close'])
    except Exception:
        return None, None

    if len(weekly) < max(fast_period, slow_period):
        return None, None

    w_sma = weekly['Close'].rolling(window=fast_period).mean()
    w_ema = weekly['Close'].ewm(span=slow_period, adjust=False).mean()

    return w_sma, w_ema


def map_weekly_to_daily(weekly_series, daily_index):
    """
    Broadcasts a weekly Series back onto a daily index so weekly lines can be
    drawn on the daily chart. Each daily bar receives its own week's value.
    """
    if weekly_series is None:
        return pd.Series([float('nan')] * len(daily_index), index=daily_index)

    try:
        w = weekly_series.copy()
        w.index = w.index.to_period('W-FRI')
        mapped = w.reindex(daily_index.to_period('W-FRI'))
        return pd.Series(mapped.values, index=daily_index)
    except Exception:
        return pd.Series([float('nan')] * len(daily_index), index=daily_index)


def detect_bullish_patterns(df):
    """Detects 5 specific bullish candlestick patterns on the latest trading day."""
    if len(df) < 2:
        return "None"
        
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    c_o, c_c, c_h, c_l = curr['Open'], curr['Close'], curr['High'], curr['Low']
    p_o, p_c, p_h, p_l = prev['Open'], prev['Close'], prev['High'], prev['Low']
    
    c_body = abs(c_c - c_o)
    c_range = c_h - c_l
    p_body = abs(p_c - p_o)
    
    patterns = []
    
    # 1. Bullish Engulfing
    if (p_c < p_o) and (c_c > c_o) and (c_o <= p_c) and (c_c >= p_o):
        patterns.append("Engulfing")
        
    # 2. Hammer
    lower_shadow = min(c_o, c_c) - c_l
    upper_shadow = c_h - max(c_o, c_c)
    if (lower_shadow >= 2 * c_body) and (upper_shadow <= c_range * 0.1) and (c_range > 0):
        patterns.append("Hammer")
        
    # 3. Bullish Harami
    if (p_c < p_o) and (c_c > c_o) and (c_o > p_c) and (c_c < p_o):
        patterns.append("Harami")
        
    # 4. Piercing Line
    if (p_c < p_o) and (c_c > c_o) and (c_o < p_c) and (c_c > (p_o + p_c) / 2):
        patterns.append("Piercing")
        
    # 5. Doji
    if c_range > 0 and c_body <= (c_range * 0.1): 
        patterns.append("Doji")
        
    return ", ".join(patterns) if patterns else "None"

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Swing Trading Dashboard", layout="wide", page_icon="📈")
st.title("Swing Trading & Pullback Screener 📈")

# ==========================================
# HELPER: FETCH ALL MARKET TICKERS (FTP FIX)
# ==========================================
@st.cache_data(ttl=86400) # Cache the ticker list for 24 hours (list rarely changes)
def get_market_tickers():
    try:
        ftp = ftplib.FTP('ftp.nasdaqtrader.com')
        ftp.login() 
        lines = []
        ftp.retrlines('RETR symboldirectory/nasdaqtraded.txt', lines.append)
        ftp.quit()
        
        data = "\n".join(lines)
        df = pd.read_csv(io.StringIO(data), sep='|')
        df = df[:-1] 
        df = df[(df['ETF'] == 'N') & (df['Test Issue'] == 'N')]
        market_stocks = df[df['Listing Exchange'].isin(['Q', 'N'])]
        tickers = [str(t).replace('.', '-') for t in market_stocks['Symbol'].tolist()]
        return tickers
    except Exception as e:
        st.error(f"Could not fetch full market list: {e}")
        return []

# ==========================================
# SIDEBAR / SETTINGS
# ==========================================
st.sidebar.header("⚙️ Screener Settings")

scan_mode = st.sidebar.radio(
    "Select Scan Mode:",
    ("Custom Watchlist", "Full Market (NYSE & NASDAQ)")
)

if scan_mode == "Custom Watchlist":
    default_tickers = "AAPL, MSFT, NVDA, TSLA, META, AMZN, GOOGL, AMD, NFLX, SPY, QQQ"
    tickers_input = st.sidebar.text_area("Tickers to Scan (comma separated)", value=default_tickers)
    tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
else:
    full_market_list = get_market_tickers()
    total_market_stocks = len(full_market_list)
    
    if total_market_stocks > 0:
        stocks_to_scan = st.sidebar.number_input(
            f"Stocks to Scan (Max: {total_market_stocks})", 
            min_value=1, 
            max_value=total_market_stocks, 
            value=min(100, total_market_stocks),
            step=50
        )
        
        tickers_list = full_market_list[:stocks_to_scan]
        
        if stocks_to_scan < total_market_stocks:
            st.sidebar.info(f"🧪 **Testing Mode:** Scanning limited to first {stocks_to_scan} stocks.")
        else:
            st.sidebar.warning(f"⚠️ **Warning:** Scanning all {total_market_stocks} stocks takes 5-10 minutes.")
    else:
        tickers_list = []
        st.sidebar.error("Failed to load market tickers.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Interactive Parameters")

fast_ma = st.sidebar.number_input("Fast SMA Period (Default 10)", min_value=3, max_value=50, value=10)
slow_ma = st.sidebar.number_input("Slow EMA Period (Default 30)", min_value=10, max_value=200, value=30)
min_price = st.sidebar.number_input("Minimum Stock Price ($)", min_value=1.0, value=5.0, step=1.0)
min_volume = st.sidebar.number_input("Min Volume (60-day EMA)", min_value=10000, value=300000, step=100000)

# NEW: directional day-over-day volume comparison
volume_direction = st.sidebar.radio(
    "Today's Volume vs Previous Day:",
    ("Off", "Greater than previous day", "Less than previous day"),
    index=0
)

# CHANGE: ADX is now a range slider
adx_min, adx_max = st.sidebar.slider("ADX Strength Range", min_value=0, max_value=100, value=(25, 100), step=1)

# NEW: Timing Filters
st.sidebar.markdown("---")
st.sidebar.subheader("⏳ Timing Filters")
use_prev_day = st.sidebar.checkbox("Scan on Previous Day's Close (Ignore Today's Live Data)", value=False)

# ==========================================
# NEW: TREND TIMEFRAME FILTERS (DAILY / WEEKLY)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Trend Filters (Daily / Weekly)")

require_daily_trend = st.sidebar.checkbox(
    f"Require Daily Trend ({fast_ma} SMA > {slow_ma} EMA)", value=True
)

require_weekly_trend = st.sidebar.checkbox(
    "Require Weekly Trend (Weekly SMA > Weekly EMA)", value=False
)

weekly_fast_ma = st.sidebar.number_input(
    "Weekly Fast SMA Period (Default 10)", min_value=2, max_value=52, value=10
)
weekly_slow_ma = st.sidebar.number_input(
    "Weekly Slow EMA Period (Default 30)", min_value=3, max_value=104, value=30
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Long-Term Trend")
# CHANGE: Baseline filter updated to Close > 200 SMA
require_200_sma = st.sidebar.checkbox("Require Close > 200 SMA (Baseline Filter)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🕯️ Candlestick Filters")

# CHANGE: Added toggle for descending highs
require_descending_highs = st.sidebar.checkbox("Require Descending Highs (3 Days)", value=False)

# CHANGE: Added toggle for the basic green candle check
require_green_candle = st.sidebar.checkbox("Require Green Candle (Close > Open)", value=True)

candlestick_filter = st.sidebar.multiselect(
    "Require Bullish Pattern (Leave blank for no pattern filter)",
    ["Engulfing", "Hammer", "Harami", "Piercing", "Doji"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Current Screening Logic:**\n"
    f"- **Daily Trend:** {f'{fast_ma} SMA > {slow_ma} EMA' if require_daily_trend else 'Off'}\n"
    f"- **Weekly Trend:** {f'{weekly_fast_ma}W SMA > {weekly_slow_ma}W EMA' if require_weekly_trend else 'Off'}\n"
    f"- **Pullback:** Close is between {fast_ma} SMA and {slow_ma} EMA\n"
    f"- **Candle:** {', '.join(candlestick_filter) if candlestick_filter else ('Green Only' if require_green_candle else 'Any Candle')}\n"
    f"- **Strength:** 10-period ADX between {adx_min} and {adx_max}\n"
    f"- **Baseline:** Close > 200 SMA (if checked)\n"
    f"- **Filters:** Price > ${min_price}, Vol > {min_volume}\n"
    f"- **Descending Highs:** {'Yes' if require_descending_highs else 'No'}\n"
    f"- **Volume vs Prev Day:** {volume_direction}"
)

# ==========================================
# DATA FETCHING
# ==========================================
# NOTE: The @st.cache_data(ttl=3600) decorator that used to sit here was the
# cause of the "stale snapshot" problem. With it, clicking "Run Screener" within
# an hour of the first run just replayed the earlier download instead of pulling
# fresh bars after the close. Freshness is now controlled explicitly by the
# button: each click performs a live download, and the result is held in
# st.session_state so that adjusting parameters recalculates indicators without
# re-downloading. Click "Run Screener" again whenever you want fresh data.
#
# LOOKBACK NOTE: the window was widened from 400 to 800 calendar days purely so
# the weekly filter has enough weekly bars (~114) for a 30-week EMA to be fully
# converged. Every daily indicator here (200 SMA, 60-EMA volume, 30 EMA, 10 ADX)
# was already fully converged at 400 days, so daily results are unchanged.
def fetch_raw_data(tickers):
    start_date = datetime.date.today() - datetime.timedelta(days=800)
    
    raw_data_dict = {}
    total_tickers = len(tickers)
    
    progress_text = "Downloading market data. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, ticker in enumerate(tickers):
        progress = int(((i + 1) / total_tickers) * 100)
        my_bar.progress(progress, text=f"Fetching {ticker} ({i+1}/{total_tickers})...")
        
        try:
            # auto_adjust=False keeps raw OHLC; end defaults to now so the latest
            # (finalized-after-close) daily bar is always included.
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
            
            if df.empty or len(df) < 200: 
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            raw_data_dict[ticker] = df.dropna()
        except Exception:
            pass 
            
    my_bar.empty() 
    return raw_data_dict

if st.sidebar.button("Run Screener"):
    # Defensively clear any cached data layers so this is always a fresh pull.
    st.cache_data.clear()
    st.session_state['raw_data'] = fetch_raw_data(tickers_list)
    st.session_state['last_fetch_time'] = datetime.datetime.now()
elif 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = {}
    st.info("👈 Click **Run Screener** in the sidebar to download data and begin.")

# ==========================================
# FRESHNESS INDICATOR
# ==========================================
raw_data = st.session_state['raw_data']

if raw_data:
    fetch_time = st.session_state.get('last_fetch_time')

    # Find the most recent bar date across everything we downloaded.
    latest_bar_date = None
    for _df in raw_data.values():
        if not _df.empty:
            d = _df.index[-1]
            if latest_bar_date is None or d > latest_bar_date:
                latest_bar_date = d

    fetch_str = fetch_time.strftime("%Y-%m-%d %H:%M:%S") if fetch_time else "unknown"
    bar_str = latest_bar_date.strftime("%Y-%m-%d") if latest_bar_date is not None else "n/a"
    st.caption(
        f"🕒 Data last downloaded: **{fetch_str}**  |  "
        f"Most recent bar in dataset: **{bar_str}**  |  "
        f"Re-click **Run Screener** to refresh."
    )

# ==========================================
# TABS SETUP
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔍 Pullback Screeners", "📊 Charting", "⏱️ Market Timing (Macro)"])

# ==========================================
# TAB 1: DYNAMIC PULLBACK SCREENER
# ==========================================
with tab1:
    st.header("Dynamic TAZ Pullback & ADX Screener")
    
    if raw_data:
        results = []
        
        with st.spinner("Calculating indicators based on your parameters..."):
            for ticker, df in raw_data.items():
                try:
                    df_calc = df.copy()

                    # Drop the live/current day if the toggle is checked
                    if use_prev_day and len(df_calc) > 2:
                        df_calc = df_calc.iloc[:-1]

                    # 1. Fast Filter: Check Price, Volume
                    latest_close = float(df_calc['Close'].iloc[-1])
                    latest_open = float(df_calc['Open'].iloc[-1])
                    latest_high = float(df_calc['High'].iloc[-1])
                    prev1_high = float(df_calc['High'].iloc[-2]) if len(df_calc) >= 2 else 0
                    prev2_high = float(df_calc['High'].iloc[-3]) if len(df_calc) >= 3 else 0
                    vol_ema = float(df_calc['Volume'].ewm(span=60, adjust=False).mean().iloc[-1])

                    # NEW: raw daily volumes for the day-over-day comparison
                    latest_volume = float(df_calc['Volume'].iloc[-1])
                    prev_volume = float(df_calc['Volume'].iloc[-2]) if len(df_calc) >= 2 else 0.0

                    if latest_close < min_price or vol_ema < min_volume:
                        continue

                    # NEW: directional volume filter
                    if volume_direction == "Greater than previous day" and not (latest_volume > prev_volume):
                        continue
                    if volume_direction == "Less than previous day" and not (latest_volume < prev_volume):
                        continue

                    # NEW: Weekly trend (10W SMA vs 30W EMA by default)
                    w_sma_series, w_ema_series = calculate_weekly_trend(
                        df_calc, weekly_fast_ma, weekly_slow_ma
                    )

                    if (w_sma_series is None) or pd.isna(w_sma_series.iloc[-1]) or pd.isna(w_ema_series.iloc[-1]):
                        w_sma_val = 0.0
                        w_ema_val = 0.0
                        weekly_uptrend = False
                    else:
                        w_sma_val = float(w_sma_series.iloc[-1])
                        w_ema_val = float(w_ema_series.iloc[-1])
                        weekly_uptrend = w_sma_val > w_ema_val

                    if require_weekly_trend and not weekly_uptrend:
                        continue

                    # 2. Calculate Indicators
                    df_calc['SMA_Fast'] = df_calc['Close'].rolling(window=fast_ma).mean()
                    df_calc['EMA_Slow'] = df_calc['Close'].ewm(span=slow_ma, adjust=False).mean()
                    df_calc['SMA_30'] = df_calc['Close'].rolling(window=30).mean() # NEW: explicitly track 30 SMA
                    df_calc['SMA_200'] = df_calc['Close'].rolling(window=200).mean()
                    
                    # Force ADX period to 10
                    df_calc = calculate_adx(df_calc, period=10)
                    
                    latest = df_calc.iloc[-1]
                    sma_f = float(latest['SMA_Fast'])
                    ema_s = float(latest['EMA_Slow'])
                    sma_30 = float(latest['SMA_30']) if not pd.isna(latest['SMA_30']) else 0
                    sma_200 = float(latest['SMA_200']) if not pd.isna(latest['SMA_200']) else 0
                    adx_val = float(latest['ADX']) if not pd.isna(latest['ADX']) else 0
                    
                    # 3. Dynamic Logic Check
                    uptrend = sma_f > ema_s
                    in_taz = (latest_close < sma_f) and (latest_close > ema_s)
                    
                    # NEW: daily trend requirement is now its own toggle
                    daily_trend_ok = uptrend if require_daily_trend else True
                    
                    # CHANGE: Use ADX min and max
                    strong_trend = adx_min <= adx_val <= adx_max
                    
                    # CHANGE: Evaluate Close vs 200 SMA
                    above_200 = latest_close > sma_200
                    if require_200_sma and not above_200:
                        continue
                        
                    # CHANGE: Evaluate Descending Highs
                    if require_descending_highs:
                        if not (latest_high < prev1_high and prev1_high < prev2_high):
                            continue

                    # 4. Candlestick Pattern Checking
                    detected_patterns_str = detect_bullish_patterns(df_calc)
                    detected_list = [p.strip() for p in detected_patterns_str.split(",")] if detected_patterns_str != "None" else []
                    
                    passes_pattern_filter = True
                    
                    # CHANGE: STRICTLY greater than
                    is_green = latest_close > latest_open
                    
                    if candlestick_filter:
                        # If user selected specific patterns, at least one must match
                        if not any(p in candlestick_filter for p in detected_list):
                            passes_pattern_filter = False
                    elif require_green_candle:
                        # If no patterns selected, fallback to green candle requirement (if toggled on)
                        if not is_green:
                            passes_pattern_filter = False
                    
                    if daily_trend_ok and in_taz and strong_trend and passes_pattern_filter:
                        results.append({
                            "Ticker": ticker,
                            "Pattern": detected_patterns_str,
                            "Close": round(latest_close, 2),
                            "Open": round(latest_open, 2),
                            f"{fast_ma} SMA": round(sma_f, 2),
                            f"{slow_ma} EMA": round(ema_s, 2),
                            f"{weekly_fast_ma}W SMA": round(w_sma_val, 2),
                            f"{weekly_slow_ma}W EMA": round(w_ema_val, 2),
                            "Weekly Trend": "Up" if weekly_uptrend else "Down/NA",
                            "ADX": round(adx_val, 2),
                            "Volume 60 EMA": f"{int(vol_ema):,}",
                            "Volume": f"{int(latest_volume):,}",
                            "Prev Volume": f"{int(prev_volume):,}",
                            "Current High": round(latest_high, 2),
                            "Prev 1 High": round(prev1_high, 2),
                            "Prev 2 High": round(prev2_high, 2)
                        })
                except Exception:
                    continue
                
        if results:
            results_df = pd.DataFrame(results)
            # Order columns nicely
            cols = ["Ticker", "Pattern", "Close", "Open", f"{fast_ma} SMA", f"{slow_ma} EMA",
                    f"{weekly_fast_ma}W SMA", f"{weekly_slow_ma}W EMA", "Weekly Trend", "ADX",
                    "Volume 60 EMA", "Volume", "Prev Volume",
                    "Current High", "Prev 1 High", "Prev 2 High"]
            # Guard against a duplicate label if daily and weekly periods collide in name
            cols = list(dict.fromkeys([c for c in cols if c in results_df.columns]))
            results_df = results_df[cols].sort_values(by="ADX", ascending=False).reset_index(drop=True)
            st.dataframe(results_df, use_container_width=True)
            st.success(f"Found {len(results_df)} setups out of {len(tickers_list)} scanned stocks.")
        else:
            st.warning("No stocks currently meet your strict parameters. Try widening the moving averages, lowering the ADX, or removing specific candlestick filters.")

# ==========================================
# TAB 2: DYNAMIC CHARTING
# ==========================================
with tab2:
    st.header("Interactive Analysis Chart")
    if raw_data:
        selected_ticker = st.selectbox("Select a ticker to view chart:", list(raw_data.keys()))
        
        df_chart = raw_data[selected_ticker].copy()
        
        # Drop the live/current day if the toggle is checked
        if use_prev_day and len(df_chart) > 2:
            df_chart = df_chart.iloc[:-1]
            
        df_chart['SMA_Fast'] = df_chart['Close'].rolling(window=fast_ma).mean()
        df_chart['EMA_Slow'] = df_chart['Close'].ewm(span=slow_ma, adjust=False).mean()
        df_chart['SMA_30'] = df_chart['Close'].rolling(window=30).mean()
        df_chart['SMA_200'] = df_chart['Close'].rolling(window=200).mean()
        
        # Force ADX period to 10
        df_chart = calculate_adx(df_chart, period=10)

        # NEW: weekly moving averages broadcast onto the daily index.
        # Computed on the FULL history before the tail() so the weekly EMA is converged.
        w_sma_chart, w_ema_chart = calculate_weekly_trend(
            df_chart, weekly_fast_ma, weekly_slow_ma
        )
        df_chart['W_SMA_Fast'] = map_weekly_to_daily(w_sma_chart, df_chart.index)
        df_chart['W_EMA_Slow'] = map_weekly_to_daily(w_ema_chart, df_chart.index)
            
        df_chart = df_chart.tail(150) 
        
        # CHANGE: Aligning setup_mask charting logic with the new filters
        setup_mask = (
            (df_chart['Close'] < df_chart['SMA_Fast']) & 
            (df_chart['Close'] > df_chart['EMA_Slow']) & 
            (df_chart['ADX'] >= adx_min) & 
            (df_chart['ADX'] <= adx_max)
        )

        # NEW: daily trend leg of the mask is now toggle-driven
        if require_daily_trend:
            setup_mask = setup_mask & (df_chart['SMA_Fast'] > df_chart['EMA_Slow'])

        # NEW: mirror the weekly trend filter on the chart markers
        if require_weekly_trend:
            setup_mask = setup_mask & (df_chart['W_SMA_Fast'] > df_chart['W_EMA_Slow'])
        
        if require_green_candle and not candlestick_filter:
            setup_mask = setup_mask & (df_chart['Close'] > df_chart['Open'])
            
        if require_200_sma:
            setup_mask = setup_mask & (df_chart['Close'] > df_chart['SMA_200'])

        # NEW: mirror the volume filter on the chart markers
        if volume_direction == "Greater than previous day":
            setup_mask = setup_mask & (df_chart['Volume'] > df_chart['Volume'].shift(1))
        elif volume_direction == "Less than previous day":
            setup_mask = setup_mask & (df_chart['Volume'] < df_chart['Volume'].shift(1))

        setup_mask = setup_mask.fillna(False)

        setup_dates = df_chart[setup_mask].index
        setup_prices = df_chart[setup_mask]['Low'] * 0.98 
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'), row=1, col=1)
        
        if not setup_dates.empty:
            fig.add_trace(go.Scatter(
                x=setup_dates, y=setup_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='DarkSlateGrey')),
                name='Setup Trigger'
            ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_Fast'], line=dict(color='blue', width=1.5), name=f'{fast_ma} SMA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_Slow'], line=dict(color='red', width=1.5), name=f'{slow_ma} EMA'), row=1, col=1)
        
        # NEW: weekly moving averages drawn as stepped lines on the daily chart
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart['W_SMA_Fast'],
            line=dict(color='#00B4D8', width=1.5, shape='hv'),
            name=f'{weekly_fast_ma}W SMA'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart['W_EMA_Slow'],
            line=dict(color='#C1121F', width=1.5, shape='hv'),
            name=f'{weekly_slow_ma}W EMA'
        ), row=1, col=1)
        
        # Adding 30 SMA to the chart so you can visualize the baseline cross
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_30'], line=dict(color='#8A2BE2', width=1, dash='dot'), name='30 SMA Baseline'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_200'], line=dict(color='#FF5F1F', width=2, dash='dot'), name='200 SMA Baseline'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ADX'], line=dict(color='purple', width=2), name='ADX (10)'), row=2, col=1)
        
        # Highlighting the ADX zone
        fig.add_hline(y=adx_min, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=adx_max, line_dash="dash", line_color="red", row=2, col=1)

        fig.update_layout(title=f'{selected_ticker} - Dynamic Technical Chart', height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Run the screener first to view charts.")

# ==========================================
# TAB 3: MARKET TIMING (MACRO)
# ==========================================
with tab3:
    st.header("Macro Market Timing")
    st.write("Using the Volatility Index (^VIX) to gauge overall market fear and greed.")
    
    try:
        vix = yf.download("^VIX", period="1y", progress=False, auto_adjust=False)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.droplevel(1)
            
        vix['SMA_10'] = vix['Close'].rolling(window=10).mean()
        
        latest_vix = float(vix.iloc[-1]['Close'])
        latest_vix_sma = float(vix.iloc[-1]['SMA_10'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current VIX", f"{latest_vix:.2f}")
            st.metric("VIX 10-Day SMA", f"{latest_vix_sma:.2f}")
            
        with col2:
            if latest_vix > (latest_vix_sma * 1.10):
                st.error("🚨 **Market Bias: CAUTION / CASH**\nVIX is spiking more than 10% above its moving average. The market is in fear.")
            elif latest_vix < latest_vix_sma:
                st.success("🟢 **Market Bias: LONG**\nVIX is cooling off below its moving average. Generally a safer environment for swing trades.")
            else:
                st.warning("🟡 **Market Bias: NEUTRAL**\nVIX is hovering near its moving average. Trade carefully.")
                
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(x=vix.index[-100:], y=vix['Close'].tail(100), name='VIX Close', line=dict(color='orange')))
        fig_vix.add_trace(go.Scatter(x=vix.index[-100:], y=vix['SMA_10'].tail(100), name='10 SMA', line=dict(color='blue', dash='dash')))
        fig_vix.update_layout(title='VIX vs 10-Period SMA (Last 100 Days)', height=400)
        st.plotly_chart(fig_vix, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load market timing data. ({e})")