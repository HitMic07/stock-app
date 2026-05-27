import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

start = '2011-01-01'
end = '2021-12-31'

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end, auto_adjust=True)
df = pd.DataFrame(df['Close'])
df.columns = ['Close']
df.dropna(inplace=True)

# Describing the data
st.subheader('Data From 2011 - 2021')
st.write(df.describe())

# Chart 1 - Closing Price
st.subheader('Closing Price vs Time Graph')
fig = plt.figure(figsize=(15, 6))
plt.plot(df.Close)
st.pyplot(fig)

# Chart 2 - 100MA
st.subheader('Closing Price vs Time 100MA Graph')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(15, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

# Chart 3 - 100MA and 200MA
st.subheader('Closing Price vs Time 100MA & 200MA Graph')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(15, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

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
final_df = pd.concat([past100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i, 0])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final chart - Predicted vs Actual
st.subheader('Predicted Price Vs Actual Price Graph')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label='Actual Price')
plt.plot(y_predicted, 'r', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
