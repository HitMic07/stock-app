import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="StockSense", page_icon="📈", layout="wide")

# ── CLEAN MINIMAL STYLE ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #ffffff; }
.main .block-container { padding: 2rem 3rem; max-width: 1200px; }
section[data-testid="stSidebar"] {
    background-color: #f9f9f9;
    border-right: 1px solid #eeeeee;
}
section[data-testid="stSidebar"] * { color: #333333 !important; }
html, body, p, span, div, label { color: #222222 !important; }
#MainMenu, footer, header { visibility: hidden; }
.stTextInput input {
    background-color: #f5f5f5 !important;
    color: #222222 !important;
    border: 1.5px solid #e0e0e0 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
}
h1 { color: #1a1a2e !important; font-weight: 700 !important; font-size: 2rem !important; }
h2, h3 { color: #1a1a2e !important; font-weight: 600 !important; }
[data-testid="metric-container"] {
    background: #f9f9f9;
    border: 1px solid #eeeeee;
    border-radius: 14px;
    padding: 20px 16px;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #888888 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
}
.pill {
    display: inline-block;
    background: #f0f0f0;
    color: #555555 !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.sig-buy  { background:#f0faf4; border:2px solid #22c55e; color:#15803d !important;
            padding:14px 24px; border-radius:14px; font-size:20px; font-weight:700;
            text-align:center; letter-spacing:2px; }
.sig-sell { background:#fff5f5; border:2px solid #ef4444; color:#b91c1c !important;
            padding:14px 24px; border-radius:14px; font-size:20px; font-weight:700;
            text-align:center; letter-spacing:2px; }
.sig-hold { background:#fffbeb; border:2px solid #f59e0b; color:#b45309 !important;
            padding:14px 24px; border-radius:14px; font-size:20px; font-weight:700;
            text-align:center; letter-spacing:2px; }
.numbox { background:#f9f9f9; border:1px solid #eeeeee; border-radius:14px;
          padding:16px; font-size:13px; color:#333 !important; line-height:2.2; }
.numbox b { color:#1a1a2e !important; }
hr { border: none; border-top: 1px solid #eeeeee; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB LIGHT THEME ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#ffffff',
    'axes.facecolor':    '#fafafa',
    'axes.edgecolor':    '#e0e0e0',
    'axes.labelcolor':   '#555555',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.color':       '#888888',
    'ytick.color':       '#888888',
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'text.color':        '#222222',
    'grid.color':        '#eeeeee',
    'grid.linestyle':    '-',
    'grid.alpha':        1,
    'legend.facecolor':  '#ffffff',
    'legend.edgecolor':  '#eeeeee',
    'legend.labelcolor': '#333333',
    'legend.fontsize':   11,
})

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 StockSense")
    st.markdown("<p style='color:#888;font-size:13px;'>AI-powered stock analysis</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#aaa;font-size:11px;line-height:1.8;'>Built by <b style='color:#555'>Omkar</b><br>Python · Random Forest<br>yfinance · Streamlit</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL CODE STARTS HERE — UNCHANGED EXCEPT ONE BUG FIX MARKED BELOW
# ══════════════════════════════════════════════════════════════════════════════

start = '2011-01-01'
end   = '2021-12-31'

st.title('Stock Price Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end, auto_adjust=True, progress=False)
df = pd.DataFrame(df['Close'])
df.columns = ['Close']
df.dropna(inplace=True)

# ── Metric cards (design addition, not original) ──────────────────────────────
st.markdown("---")
current  = float(df['Close'].iloc[-1])
start_p  = float(df['Close'].iloc[0])
ret_pct  = ((current - start_p) / start_p) * 100
highest  = float(df['Close'].max())
lowest   = float(df['Close'].min())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price",  f"${current:.2f}")
c2.metric("Total Return",   f"{ret_pct:.1f}%", delta=f"{ret_pct:.1f}%")
c3.metric("All-Time High",  f"${highest:.2f}")
c4.metric("All-Time Low",   f"${lowest:.2f}")
st.markdown("---")

# Describing the data
st.subheader('Data From 2011 - 2021')
st.write(df.describe())

# Chart 1 - Closing Price
st.markdown('<div class="pill">Price History</div>', unsafe_allow_html=True)
st.subheader('Closing Price vs Time Graph')
fig = plt.figure(figsize=(15, 6))
plt.plot(df.Close, color='#1a1a2e', linewidth=1.5)
plt.fill_between(range(len(df)), df.Close.values, alpha=0.06, color='#1a1a2e')
plt.grid(True)
st.pyplot(fig)

# Chart 2 - 100MA
st.markdown('<div class="pill">Trend</div>', unsafe_allow_html=True)
st.subheader('Closing Price vs Time 100MA Graph')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(15, 6))
plt.plot(df.Close, color='#cccccc', linewidth=1.2, label='Close Price')
plt.plot(ma100, color='#f59e0b', linewidth=2, label='100-Day MA')
plt.legend()
plt.grid(True)
st.pyplot(fig)

# Chart 3 - 100MA and 200MA
st.subheader('Closing Price vs Time 100MA & 200MA Graph')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(15, 6))
plt.plot(df.Close, color='#cccccc', linewidth=1.2, label='Close Price')
plt.plot(ma100, color='#f59e0b', linewidth=2, label='100-Day MA')
plt.plot(ma200, color='#1a1a2e', linewidth=2, label='200-Day MA')
plt.legend()
plt.grid(True)
st.pyplot(fig)

st.markdown("---")

# 70% training, 30% testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing  = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Build training sequences
x_train, y_train = [], []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i, 0])
    y_train.append(data_training_array[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Train model
with st.spinner('Training model... please wait'):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

# Testing
past100_days = data_training.tail(100)
final_df     = pd.concat([past100_days, data_testing], ignore_index=True)

# ▼▼▼ THE ONE BUG FIX: changed fit_transform → transform (scaler already fitted above)
input_data = scaler.transform(final_df)
# ▲▲▲

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i, 0])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted  = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted  = y_predicted * scale_factor
y_test       = y_test      * scale_factor

# Final chart - Predicted vs Actual
st.markdown('<div class="pill">AI Prediction</div>', unsafe_allow_html=True)
st.subheader('Predicted Price Vs Actual Price Graph')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test,      color='#1a1a2e', linewidth=1.5, label='Actual Price')
plt.plot(y_predicted, color='#f59e0b', linewidth=1.5, label='Predicted', linestyle='--')
plt.fill_between(range(len(y_test)), y_test, y_predicted, alpha=0.05, color='#f59e0b')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
st.pyplot(fig2)

# ── Buy/Sell Signal (design addition) ────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="pill">Signal</div>', unsafe_allow_html=True)
st.markdown("### AI Buy / Sell Signal")

last_actual    = float(y_test[-1])
last_predicted = float(y_predicted[-1])
diff_pct       = ((last_predicted - last_actual) / last_actual) * 100
ma100_last     = float(ma100.dropna().iloc[-1])
ma200_last     = float(ma200.dropna().iloc[-1])

cs1, cs2, cs3 = st.columns(3)

with cs1:
    st.markdown("**Model Signal**")
    if diff_pct > 1.5:
        st.markdown('<div class="sig-buy">▲ BUY</div>', unsafe_allow_html=True)
        st.caption("Model predicts price will rise")
    elif diff_pct < -1.5:
        st.markdown('<div class="sig-sell">▼ SELL</div>', unsafe_allow_html=True)
        st.caption("Model predicts price will fall")
    else:
        st.markdown('<div class="sig-hold">— HOLD</div>', unsafe_allow_html=True)
        st.caption("No strong directional signal")

with cs2:
    st.markdown("**MA Crossover**")
    if ma100_last > ma200_last:
        st.markdown('<div class="sig-buy">▲ BULLISH</div>', unsafe_allow_html=True)
        st.caption("100MA above 200MA — uptrend")
    else:
        st.markdown('<div class="sig-sell">▼ BEARISH</div>', unsafe_allow_html=True)
        st.caption("100MA below 200MA — downtrend")

with cs3:
    st.markdown("**Key Numbers**")
    st.markdown(f"""
    <div class="numbox">
    Last Actual &nbsp;&nbsp;&nbsp;&nbsp; <b>${last_actual:.2f}</b><br>
    AI Predicted &nbsp; <b>${last_predicted:.2f}</b><br>
    Difference &nbsp;&nbsp;&nbsp;&nbsp; <b>{diff_pct:+.2f}%</b><br>
    100-Day MA &nbsp;&nbsp; <b>${ma100_last:.2f}</b><br>
    200-Day MA &nbsp;&nbsp; <b>${ma200_last:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.warning("⚠️ Educational purposes only. Not financial advice.")
st.markdown("<p style='text-align:center;color:#aaa;font-size:12px;'>Built by Omkar · Python · Random Forest · yfinance · Streamlit</p>", unsafe_allow_html=True)
