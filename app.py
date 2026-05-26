import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Stock Trend Predictor", layout="centered")
st.title('📈 Stock Trend Prediction App')
st.markdown("Built with **Random Forest ML** + **Streamlit**")

user_input = st.text_input('🔍 Enter Stock Ticker Symbol', 'SBIN.NS')

raw = yf.download(user_input, start='2010-01-01', end='2023-12-31', auto_adjust=True)

if raw.empty:
    st.error("Could not fetch data. Check ticker and try again.")
    st.stop()

# Fix MultiIndex columns from new yfinance
raw.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in raw.columns]

# Find the Close column regardless of name
close_col = [c for c in raw.columns if 'Close' in c or 'close' in c][0]
df = pd.DataFrame()
df['Close'] = raw[close_col].values.astype(float)
df.dropna(inplace=True)

st.subheader(f'Data for {user_input} (2010–2023)')
st.write(df.describe())

st.subheader('Closing Price Over Time')
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(df['Close'].values, label='Closing Price', color='blue')
ax1.legend()
st.pyplot(fig1)

ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

st.subheader('100-Day & 200-Day Moving Average')
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(df['Close'].values, label='Closing Price', color='blue')
ax2.plot(ma100.values, 'r', label='100-Day MA')
ax2.plot(ma200.values, 'g', label='200-Day MA')
ax2.legend()
st.pyplot(fig2)

st.subheader('Predicted Price vs Original Price')

prices = df['Close'].values.flatten().astype(float)
prices = prices.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(prices)

X, y = [], []
for i in range(100, len(scaled)):
    X.append(scaled[i-100:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

split = int(len(X) * 0.70)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

with st.spinner('Training model... please wait ~30 seconds'):
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

print("y_test_real sample:", y_test_real[:5])
print("y_pred_real sample:", y_pred_real[:5])

fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.plot(y_test_real, 'b', label='Original Price')
ax3.plot(y_pred_real, 'r', label='Predicted Price')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Stock Price (₹)')
ax3.legend()
st.pyplot(fig3)

st.success("✅ Prediction complete! The closer the red line is to blue, the better.")
st.markdown("---")
st.markdown("Built with Random Forest · yfinance · scikit-learn · Streamlit")
