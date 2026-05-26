import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Stock Trend Predictor", layout="centered")
st.title('📈 Stock Trend Prediction App')

user_input = st.text_input('🔍 Enter Stock Ticker Symbol', 'SBIN.NS')

raw = yf.download(user_input, start='2010-01-01', end='2023-12-31', auto_adjust=True)

if raw.empty:
    st.error("Could not fetch data.")
    st.stop()

# Flatten whatever yfinance returns into a simple 1D array
try:
    prices = raw['Close'].values.flatten().astype(float)
except:
    prices = raw.iloc[:, 0].values.flatten().astype(float)

st.write("DEBUG — First 5 prices:", prices[:5])
st.write("DEBUG — Last 5 prices:", prices[-5:])
st.write("DEBUG — Min price:", prices.min(), "Max price:", prices.max())
st.write("DEBUG — Total rows:", len(prices))
