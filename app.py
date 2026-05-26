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
    st.error("Could not fetch data.")
    st.stop()

# Get close prices as clean 1D numpy array
try:
    prices = raw['Close'].values.flatten().astype(float)
except:
    prices = raw.iloc[:, 0].values.flatten().astype(float)

prices = prices[~np.isnan(prices)]  # remove any NaN values

# ---- CHARTS ----

st.subheader('Closing Price Over Time')
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(prices, label='Closing Price', color='blue')
ax1.legend()
st.pyplot(fig1)

ma100 = pd.Series(prices).rolling(100).mean().values
ma200 = pd.Series(prices).rolling(200).mean().values

st.subheader('100-Day & 200-Day Moving Average')
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(prices, label='Closing Price', color='blue')
ax2.plot(ma100, 'r', label='100-Day MA')
ax2.plot(ma200, 'g', label='200-Day MA')
ax2.legend()
st.pyplot(fig2)

# ---- ML MODEL ----
st.subheader('Predicted Price vs Original Price')

# Split raw prices FIRST before any scaling
split = int(len(prices) * 0.70)
train_prices = prices[:split]
test_prices  = prices[split:]

# Fit scaler ONLY on training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_prices.reshape(-1, 1))

# Scale all prices using that scaler
all_scaled = scaler.transform(prices.reshape(-1, 1)).flatten()

# Build sequences
X, y = [], []
for i in range(100, len(all_scaled)):
    X.append(all_scaled[i-100:i])
    y.append(all_scaled[i])

X = np.array(X)
y = np.array(y)

# Split sequences same way
X_train = X[:split-100]
y_train = y[:split-100]
X_test  = X[split-100:]
y_test  = y[split-100:]

# Train
with st.spinner('Training model... please wait ~30 seconds'):
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Inverse transform back to real prices
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

st.success("✅ Prediction complete! The closer red is to blue, the better the model.")
st.markdown("---")
st.markdown("Built with Random Forest · yfinance · scikit-learn · Streamlit")
