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

    df['TR'] = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)

    df['+DM'] = (df['High'] - df['High'].shift()).clip(lower=0)
    df['-DM'] = (df['Low'].shift() - df['Low']).clip(lower=0)

    df['+DM'] = df['+DM'].where(df['+DM'] > df['-DM'], 0)
    df['-DM'] = df['-DM'].where(df['-DM'] > df['+DM'], 0)

    tr_smooth = df['TR'].rolling(period).mean()
    plus_dm_smooth = df['+DM'].rolling(period).mean()
    minus_dm_smooth = df['-DM'].rolling(period).mean()

    df['+DI'] = 100 * (plus_dm_smooth / tr_smooth)
    df['-DI'] = 100 * (minus_dm_smooth / tr_smooth)

    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(period).mean()

    return df

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
@st.cache_data(ttl=86400) # Cache the ticker list for 24 hours
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
min_volume = st.sidebar.number_input("Min Average Volume (20-day)", min_value=10000, value=500000, step=100000)

# CHANGE: ADX is now a range slider
adx_min, adx_max = st.sidebar.slider("ADX Strength Range", min_value=0, max_value=100, value=(25, 100), step=1)

# NEW: Timing Filters
st.sidebar.markdown("---")
st.sidebar.subheader("⏳ Timing Filters")
use_prev_day = st.sidebar.checkbox("Scan on Previous Day's Close (Ignore Today's Live Data)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Long-Term Trend")
# CHANGE: Baseline filter updated to 30 SMA > 200 SMA
require_200_sma = st.sidebar.checkbox("Require 30 SMA > 200 SMA (Baseline Filter)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🕯️ Candlestick Filters")

# CHANGE: Added toggle for the basic green candle check
require_green_candle = st.sidebar.checkbox("Require Green Candle (Close > Open)", value=True)

candlestick_filter = st.sidebar.multiselect(
    "Require Bullish Pattern (Leave blank for no pattern filter)",
    ["Engulfing", "Hammer", "Harami", "Piercing", "Doji"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Current Screening Logic:**\n"
    f"- **Trend:** {fast_ma} SMA > {slow_ma} EMA\n"
    f"- **Pullback:** Close is between {fast_ma} SMA and {slow_ma} EMA\n"
    f"- **Candle:** {', '.join(candlestick_filter) if candlestick_filter else ('Green Only' if require_green_candle else 'Any Candle')}\n"
    f"- **Strength:** 10-period ADX between {adx_min} and {adx_max}\n"
    f"- **Baseline:** 30 SMA > 200 SMA (if checked)\n"
    f"- **Filters:** Price > ${min_price}, Vol > {min_volume}"
)

# ==========================================
# DATA FETCHING
# ==========================================
@st.cache_data(ttl=3600) 
def fetch_raw_data(tickers):
    start_date = datetime.date.today() - datetime.timedelta(days=400) 
    
    raw_data_dict = {}
    total_tickers = len(tickers)
    
    progress_text = "Downloading market data. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, ticker in enumerate(tickers):
        progress = int(((i + 1) / total_tickers) * 100)
        my_bar.progress(progress, text=f"Fetching {ticker} ({i+1}/{total_tickers})...")
        
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            
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
    st.session_state['raw_data'] = fetch_raw_data(tickers_list)
elif 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = {}
    st.info("👈 Click **Run Screener** in the sidebar to download data and begin.")

# ==========================================
# TABS SETUP
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔍 Pullback Screeners", "📊 Charting", "⏱️ Market Timing (Macro)"])
raw_data = st.session_state['raw_data']

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
                    avg_vol = float(df_calc['Volume'].tail(20).mean())
                    
                    if latest_close < min_price or avg_vol < min_volume:
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
                    
                    # CHANGE: Use ADX min and max
                    strong_trend = adx_min <= adx_val <= adx_max
                    
                    # CHANGE: Evaluate 30 SMA vs 200 SMA
                    above_200 = sma_30 > sma_200
                    if require_200_sma and not above_200:
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
                    
                    if uptrend and in_taz and strong_trend and passes_pattern_filter:
                        results.append({
                            "Ticker": ticker,
                            "Pattern": detected_patterns_str,
                            "Close": round(latest_close, 2),
                            "Open": round(latest_open, 2),
                            f"{fast_ma} SMA": round(sma_f, 2),
                            f"{slow_ma} EMA": round(ema_s, 2),
                            "ADX": round(adx_val, 2),
                            "Avg Volume": f"{int(avg_vol):,}"
                        })
                except Exception:
                    continue
                
        if results:
            results_df = pd.DataFrame(results)
            # Order columns nicely
            cols = ["Ticker", "Pattern", "Close", "Open", f"{fast_ma} SMA", f"{slow_ma} EMA", "ADX", "Avg Volume"]
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
            
        df_chart = df_chart.tail(150) 
        
        # CHANGE: Aligning setup_mask charting logic with the new filters
        setup_mask = (
            (df_chart['SMA_Fast'] > df_chart['EMA_Slow']) & 
            (df_chart['Close'] < df_chart['SMA_Fast']) & 
            (df_chart['Close'] > df_chart['EMA_Slow']) & 
            (df_chart['ADX'] >= adx_min) & 
            (df_chart['ADX'] <= adx_max)
        )
        
        if require_green_candle and not candlestick_filter:
            setup_mask = setup_mask & (df_chart['Close'] > df_chart['Open'])
            
        if require_200_sma:
            setup_mask = setup_mask & (df_chart['SMA_30'] > df_chart['SMA_200'])
            
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
        vix = yf.download("^VIX", period="1y", progress=False)
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