import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense — AI Prediction",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0f1e; color: #e8eaf6; }
    section[data-testid="stSidebar"] {
        background-color: #0d1b2a;
        border-right: 1px solid #1e3a5f;
    }
    html, body, [class*="css"] { color: #e8eaf6; font-family: 'Segoe UI', sans-serif; }
    [data-testid="metric-container"] {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px;
    }
    .stTextInput input {
        background-color: #0d1b2a !important;
        color: #e8eaf6 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
    }
    h1 { color: #4fc3f7 !important; font-size: 2.2rem !important; }
    h2, h3 { color: #81d4fa !important; }
    hr { border-color: #1e3a5f; }
    .section-tag {
        display: inline-block;
        background: #1e3a5f;
        color: #4fc3f7;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .signal-buy {
        background: #0d2b1e; border: 2px solid #00e676; color: #00e676;
        padding: 12px 28px; border-radius: 12px; font-size: 22px;
        font-weight: 700; text-align: center; letter-spacing: 2px;
    }
    .signal-sell {
        background: #2b0d0d; border: 2px solid #ff5252; color: #ff5252;
        padding: 12px 28px; border-radius: 12px; font-size: 22px;
        font-weight: 700; text-align: center; letter-spacing: 2px;
    }
    .signal-hold {
        background: #2b2200; border: 2px solid #ffd740; color: #ffd740;
        padding: 12px 28px; border-radius: 12px; font-size: 22px;
        font-weight: 700; text-align: center; letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0a0f1e',
    'axes.facecolor':    '#0d1b2a',
    'axes.edgecolor':    '#1e3a5f',
    'axes.labelcolor':   '#81d4fa',
    'xtick.color':       '#81d4fa',
    'ytick.color':       '#81d4fa',
    'text.color':        '#e8eaf6',
    'grid.color':        '#1e3a5f',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'legend.facecolor':  '#0d1b2a',
    'legend.edgecolor':  '#1e3a5f',
    'legend.labelcolor': '#e8eaf6',
})

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    user_input = st.text_input('🔍 Stock Ticker', 'AAPL',
                               help="AAPL, TSLA, SBIN.NS, BTC-USD etc.")
    start_year = st.selectbox("Start Year", [2010, 2012, 2015, 2018], index=0)
    end_year   = st.selectbox("End Year",   [2021, 2022, 2023, 2024], index=2)
    n_trees    = st.slider("Model Trees (higher = smarter but slower)", 50, 300, 100, 50)
    st.markdown("---")
    st.markdown("""
    <div style='color:#4fc3f7; font-size:12px;'>
    Built by <b>Omkar</b><br>
    Stack: Python · Random Forest<br>
    yfinance · scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)

start = f'{start_year}-01-01'
end   = f'{end_year}-12-31'

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 📈 StockSense — AI Price Predictor")
st.markdown(f"<p style='color:#81d4fa; font-size:16px;'>Analysing <b>{user_input}</b> from {start_year} to {end_year} using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
with st.spinner(f'Fetching data for {user_input}...'):
    raw = yf.download(user_input, start=start, end=end, auto_adjust=True)

if raw.empty:
    st.error("Could not fetch data. Check your ticker symbol.")
    st.stop()

df = pd.DataFrame()
df['Close']  = raw['Close'].values.flatten().astype(float)
df['Volume'] = raw['Volume'].values.flatten().astype(float)
df.dropna(inplace=True)

# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
current_price = df['Close'].iloc[-1]
start_price   = df['Close'].iloc[0]
change_pct    = ((current_price - start_price) / start_price) * 100
highest       = df['Close'].max()
lowest        = df['Close'].min()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Current Price",  f"${current_price:.2f}")
col2.metric("📊 Total Return",   f"{change_pct:.1f}%", delta=f"{change_pct:.1f}%")
col3.metric("🔺 All-Time High",  f"${highest:.2f}")
col4.metric("🔻 All-Time Low",   f"${lowest:.2f}")

st.markdown("---")

# ─────────────────────────────────────────────
# CHART 1 — CLOSING PRICE
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">Price History</div>', unsafe_allow_html=True)
st.markdown("### Closing Price Over Time")

fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df['Close'].values, color='#4fc3f7', linewidth=1.5, label='Close Price')
ax1.fill_between(range(len(df)), df['Close'].values, alpha=0.08, color='#4fc3f7')
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Price')
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# ─────────────────────────────────────────────
# CHART 2 — MOVING AVERAGES
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">Trend Analysis</div>', unsafe_allow_html=True)
st.markdown("### Price with 100-Day & 200-Day Moving Averages")

ma100 = pd.Series(df['Close'].values).rolling(100).mean()
ma200 = pd.Series(df['Close'].values).rolling(200).mean()

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(df['Close'].values, color='#4fc3f7', linewidth=1.2, label='Close Price', alpha=0.8)
ax2.plot(ma100.values, color='#ff9800', linewidth=1.8, label='100-Day MA')
ax2.plot(ma200.values, color='#ab47bc', linewidth=1.8, label='200-Day MA')
ax2.fill_between(range(len(df)), df['Close'].values, alpha=0.05, color='#4fc3f7')
ax2.set_xlabel('Trading Days')
ax2.set_ylabel('Price')
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# ─────────────────────────────────────────────
# CHART 3 — VOLUME
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">Volume Analysis</div>', unsafe_allow_html=True)
st.markdown("### Trading Volume Over Time")

fig3, ax3 = plt.subplots(figsize=(14, 4))
colors = ['#00e676' if df['Close'].values[i] >= df['Close'].values[i-1]
          else '#ff5252' for i in range(1, len(df))]
colors.insert(0, '#00e676')
ax3.bar(range(len(df)), df['Volume'].values, color=colors, alpha=0.7, width=1)
ax3.set_xlabel('Trading Days')
ax3.set_ylabel('Volume')
ax3.grid(True, axis='y')
green_patch = mpatches.Patch(color='#00e676', label='Up Day')
red_patch   = mpatches.Patch(color='#ff5252', label='Down Day')
ax3.legend(handles=[green_patch, red_patch])
st.pyplot(fig3)

st.markdown("---")

# ─────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">AI Prediction</div>', unsafe_allow_html=True)
st.markdown("### Predicted Price vs Actual Price")

prices = df['Close'].values.flatten().astype(float)
split  = int(len(prices) * 0.70)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(prices[:split].reshape(-1, 1))
all_scaled = scaler.transform(prices.reshape(-1, 1)).flatten()

X, y = [], []
for i in range(100, len(all_scaled)):
    X.append(all_scaled[i-100:i])
    y.append(all_scaled[i])
X, y = np.array(X), np.array(y)

X_train, y_train = X[:split-100], y[:split-100]
X_test,  y_test  = X[split-100:], y[split-100:]

with st.spinner(f'Training AI model with {n_trees} trees... please wait'):
    model = RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

y_pred      = model.predict(X_test)
y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

fig4, ax4 = plt.subplots(figsize=(14, 6))
ax4.plot(y_test_real, color='#4fc3f7', linewidth=1.5, label='Actual Price')
ax4.plot(y_pred_real, color='#ff9800', linewidth=1.5, label='Predicted Price', linestyle='--')
ax4.fill_between(range(len(y_test_real)), y_test_real, y_pred_real,
                 alpha=0.08, color='#ff9800')
ax4.set_xlabel('Trading Days (Test Period)')
ax4.set_ylabel('Price')
ax4.grid(True)
ax4.legend()
st.pyplot(fig4)

# ─────────────────────────────────────────────
# BUY / SELL / HOLD SIGNAL
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-tag">Signal</div>', unsafe_allow_html=True)
st.markdown("### AI Buy / Sell Signal")

last_actual    = y_test_real[-1]
last_predicted = y_pred_real[-1]
diff_pct       = ((last_predicted - last_actual) / last_actual) * 100
ma100_last     = ma100.dropna().iloc[-1]
ma200_last     = ma200.dropna().iloc[-1]

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    st.markdown("**Model Prediction**")
    if diff_pct > 1.5:
        st.markdown('<div class="signal-buy">🟢 BUY</div>', unsafe_allow_html=True)
        st.caption("Model predicts price will rise")
    elif diff_pct < -1.5:
        st.markdown('<div class="signal-sell">🔴 SELL</div>', unsafe_allow_html=True)
        st.caption("Model predicts price will fall")
    else:
        st.markdown('<div class="signal-hold">🟡 HOLD</div>', unsafe_allow_html=True)
        st.caption("No strong signal detected")

with col_s2:
    st.markdown("**MA Crossover Signal**")
    if ma100_last > ma200_last:
        st.markdown('<div class="signal-buy">🟢 BULLISH</div>', unsafe_allow_html=True)
        st.caption("100MA above 200MA — uptrend")
    else:
        st.markdown('<div class="signal-sell">🔴 BEARISH</div>', unsafe_allow_html=True)
        st.caption("100MA below 200MA — downtrend")

with col_s3:
    st.markdown("**Key Numbers**")
    st.markdown(f"""
    <div style='background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px; padding:12px; font-size:13px;'>
    📍 Last Actual: <b>${last_actual:.2f}</b><br>
    🤖 AI Predicted: <b>${last_predicted:.2f}</b><br>
    📊 Difference: <b>{diff_pct:+.2f}%</b><br>
    📈 100-Day MA: <b>${ma100_last:.2f}</b><br>
    📉 200-Day MA: <b>${ma200_last:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.warning("⚠️ This app is for educational purposes only. Not financial advice. Always do your own research before investing.")
st.markdown("""
<div style='text-align:center; color:#4fc3f7; font-size:12px; padding-top:10px;'>
Built by Omkar · Python · Random Forest · yfinance · Streamlit
</div>
""", unsafe_allow_html=True)
