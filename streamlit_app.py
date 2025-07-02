import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finta import TA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
import zipfile
import os

# App settings
st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📊 NYSE Stock Price Prediction (GRU + LSTM)")

# Stock selector
symbol = st.selectbox("Select Company", ['YHOO', 'MSFT', 'ADBE', 'XRX'])

# Load and extract data
@st.cache_data
def load_data():
    if os.path.exists("prices.zip"):
        with zipfile.ZipFile("prices.zip", 'r') as zip_ref:
            zip_ref.extractall()
    df = pd.read_csv("prices.csv")
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['date'])
    return df

df = load_data()

# Filter selected stock
stock_df = df[df['symbol'] == symbol].copy().sort_values('date')
stock_df.rename(columns={
    'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW',
    'close': 'CLOSE', 'volume': 'VOLUME'
}, inplace=True)

# Add technical indicators
stock_df['SMA_14'] = TA.SMA(stock_df, 14)
stock_df['EMA_14'] = TA.EMA(stock_df, 14)
stock_df['RSI_14'] = TA.RSI(stock_df)
macd = TA.MACD(stock_df)
stock_df['MACD'] = macd['MACD']
stock_df['SIGNAL'] = macd['SIGNAL']

# Drop missing values
model_df = stock_df[['CLOSE', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'SIGNAL']].dropna()
if model_df.empty or len(model_df) < 30:
    st.warning("Not enough data for this stock.")
    st.stop()

# Scaling and sequence preparation
scaler = MinMaxScaler()
scaled = scaler.fit_transform(model_df)

X, Y = [], []
n = 10  # sequence length
for i in range(n, len(scaled)):
    X.append(scaled[i-n:i])
    Y.append(scaled[i][0])

X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Build model
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)

# Predict and inverse scale
Y_pred = model.predict(X_test)

close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
Y_test_inv = close_scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_pred_inv = close_scaler.inverse_transform(Y_pred)

# Plot results
st.subheader(f"{symbol} — Predicted vs Actual Price")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(Y_test_inv, label='Actual', color='blue')
ax.plot(Y_pred_inv, label='Predicted', color='red')
ax.legend()
st.pyplot(fig)
