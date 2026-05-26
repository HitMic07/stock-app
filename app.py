 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
 
# ============================================================
# Page config — sets the browser tab title
# ============================================================
 
st.set_page_config(page_title="Stock Trend Predictor", layout="centered")
 
# ============================================================
# App title and description
# ============================================================
 
st.title('📈 Stock Trend Prediction App')
st.markdown("""
Built with **LSTM Neural Network** + **Streamlit**  
Enter any stock ticker below to see predictions vs actual prices.
""")
 
# ============================================================
# User input — stock ticker text box
# ============================================================
 
user_input = st.text_input(
    '🔍 Enter Stock Ticker Symbol',
    'SBIN.NS',
    help="Examples: AAPL, TSLA, GOOGL, RELIANCE.NS, SBIN.NS"
)
 
# ============================================================
# Download stock data for the entered ticker
# ============================================================
 
st.subheader(f'Data for {user_input} (2010 – 2023)')
 
df = yf.download(user_input, start='2010-01-01', end='2023-12-31')
 
if df.empty:
    st.error("Could not fetch data. Check your ticker symbol and try again.")
    st.stop()
 
# Show summary stats
st.write(df.describe())
 
# ============================================================
# Chart 1 — Closing Price over time
# ============================================================
 
st.subheader('Closing Price Over Time')
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df['Close'], label='Closing Price', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
st.pyplot(fig1)
 
# ============================================================
# Chart 2 — Closing Price + 100-Day Moving Average
# ============================================================
 
st.subheader('Closing Price vs 100-Day Moving Average')
ma100 = df['Close'].rolling(100).mean()
 
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df['Close'], label='Closing Price', color='blue')
ax2.plot(ma100, 'r', label='100-Day MA')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)
 
# ============================================================
# Chart 3 — Closing Price + 100-Day MA + 200-Day MA
# ============================================================
 
st.subheader('Closing Price vs 100-Day MA vs 200-Day MA')
ma200 = df['Close'].rolling(200).mean()
 
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(df['Close'], label='Closing Price', color='blue')
ax3.plot(ma100, 'r', label='100-Day MA')
ax3.plot(ma200, 'g', label='200-Day MA')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.legend()
st.pyplot(fig3)
 
# ============================================================
# Prepare test data for prediction
# ============================================================
 
# Split: 70% train, 30% test
data_training = pd.DataFrame(df['Close'][0: int(len(df) * 0.70)])
data_testing  = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])
 
scaler = MinMaxScaler(feature_range=(0, 1))
 
# Fit scaler on training data only
data_training_array = scaler.fit_transform(data_training)
 
# Combine last 100 days of training data + all test data
past_100_days = data_training.tail(100)
final_df      = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data    = scaler.fit_transform(final_df)
 
# Build test sequences (same 100-day window logic as training)
x_test = []
y_test = []
 
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
 
x_test, y_test = np.array(x_test), np.array(y_test)
 
# ============================================================
# Load the saved model and make predictions
# ============================================================
 
try:
    model = load_model('keras_model.h5')
except Exception as e:
    st.error(f"Could not load model: {e}\n\nMake sure 'keras_model.h5' is in the same folder as app.py")
    st.stop()
 
y_predicted = model.predict(x_test)
 
# Scale back to real rupee/dollar values
scale_factor = 1 / scaler.scale_[0]
y_predicted  = y_predicted * scale_factor
y_test       = y_test      * scale_factor
 
# ============================================================
# Chart 4 — Predicted vs Actual price (the money shot)
# ============================================================
 
st.subheader('Predicted Price vs Original Price')
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(y_test,      'b', label='Original Price')
ax4.plot(y_predicted, 'r', label='Predicted Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)
 
st.success("✅ Prediction complete! The closer the red line is to the blue, the better the model.")
 
# ============================================================
# Footer
# ============================================================
 
st.markdown("---")
st.markdown("Built with LSTM · yfinance · Keras · Streamlit")
 
