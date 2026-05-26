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

df = yf.download(user_input, start='2010-01-01', end='2023-12-31', auto_adjust=True)

if df.empty:
    st.error("Could not fetch data. Check ticker and try again.")
    st.stop()

df = df[['Close']].copy()
df.dropna(inplace=True)

st.subheader(f'Data for {user_input} (2010–2023)')
st.write(df.describe())

st.subheader('Closing Price Over Time')
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(df['Close'], label='Closing Price', color='blue')
ax1.legend()
st.pyplot(fig1)

ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

st.subheader('100-Day & 200-Day Moving Average')
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(df['Close'], label='Closing Price', color='blue')
ax2.plot(ma100, 'r', label='100-Day MA')
ax2.plot(ma200, 'g', label='200-Day MA')
ax2.legend()
st.pyplot(fig2)

st.subheader('Predicted Price vs Original Price')

# Get raw close prices as numpy array
prices = df['Close'].values.reshape(-1, 1)

# Scale to 0-1
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

# Build sequences of 100 days
X, y = [], []
for i in range(100, len(scaled)):
    X.append(scaled[i-100:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

# Train/test split - 70% train, 30% test
split = int(len(X) * 0.70)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
with st.spinner('Training model... please wait ~30 seconds'):
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Convert back to real prices
y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Plot
fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.plot(y_test_real, 'b', label='Original Price')
ax3.plot(y_pred_real, 'r', label='Predicted Price')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Stock Price')
ax3.legend()
st.pyplot(fig3)

st.success("✅ Prediction complete! The closer the red line is to blue, the better.")
st.markdown("---")
st.markdown("Built with Random Forest · yfinance · scikit-learn · Streamlit")
