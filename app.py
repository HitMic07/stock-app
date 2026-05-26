 
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Stock Trend Predictor", layout="centered")
st.title('📈 Stock Trend Prediction App')
st.markdown("Built with **Random Forest ML** + **Streamlit** — Enter any stock ticker below.")

user_input = st.text_input('🔍 Enter Stock Ticker Symbol', 'SBIN.NS')

df = yf.download(user_input, start='2010-01-01', end='2023-12-31')

if df.empty:
    st.error("Could not fetch data. Check ticker and try again.")
    st.stop()

st.subheader(f'Data for {user_input} (2010–2023)')
st.write(df.describe())

st.subheader('Closing Price Over Time')
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(df['Close'], label='Closing Price')
ax1.legend()
st.pyplot(fig1)

ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

st.subheader('100-Day & 200-Day Moving Average')
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(df['Close'], label='Closing Price')
ax2.plot(ma100, 'r', label='100-Day MA')
ax2.plot(ma200, 'g', label='200-Day MA')
ax2.legend()
st.pyplot(fig2)

st.subheader('Predicted Price vs Original Price')

close = df[['Close']].copy()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

X, y = [], []
for i in range(100, len(scaled)):
    X.append(scaled[i-100:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
split = int(len(X) * 0.70)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred_real = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))

fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.plot(y_test_real, 'b', label='Original Price')
ax3.plot(y_pred_real, 'r', label='Predicted Price')
ax3.legend()
st.pyplot(fig3)

st.success("✅ Prediction complete!")
st.markdown("---")
st.markdown("Built with Random Forest · yfinance · scikit-learn · Streamlit")
