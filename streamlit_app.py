import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finta import TA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout

st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📊 Stock Price Prediction (Upload Your CSV)")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your stock CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a stock price CSV file to begin.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)
st.success("File uploaded successfully!")

# Date parsing
try:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
except:
    st.error("❌ Couldn't parse 'date' column. Please make sure it's named 'date'.")
    st.stop()

# Rename columns if needed
df.rename(columns={
    'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW',
    'close': 'CLOSE', 'volume': 'VOLUME'
}, inplace=True)

required_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
if not all(col in df.columns for col in required_cols):
    st.error("❌ Required columns missing. CSV must have: open, high, low, close, volume, date")
    st.stop()

# Sort by date
df = df.sort_values('date')

# Indicators
df['SMA_14'] = TA.SMA(df, 14)
df['EMA_14'] = TA.EMA(df, 14)
df['RSI_14'] = TA.RSI(df)
macd = TA.MACD(df)
df['MACD'] = macd['MACD']
df['SIGNAL'] = macd['SIGNAL']

# Final features
model_df = df[['CLOSE', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'SIGNAL']].dropna()
if len(model_df) < 50:
    st.error("❌ Not enough data after indicators. Minimum 50 rows required.")
    st.stop()

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(model_df)

X, Y = [], []
n = 10
for i in range(n, len(scaled_data)):
    X.append(scaled_data[i-n:i])
    Y.append(scaled_data[i][0])

X, Y = np.array(X), np.array(Y)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Model
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)

# Predict
Y_pred = model.predict(X_test)
close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
Y_test_inv = close_scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_pred_inv = close_scaler.inverse_transform(Y_pred)

# Plot
st.subheader("📈 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(Y_test_inv, label='Actual', color='blue')
ax.plot(Y_pred_inv, label='Predicted', color='red')
ax.legend()
st.pyplot(fig)
