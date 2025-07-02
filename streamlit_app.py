import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finta import TA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout

st.set_page_config(page_title="📈 Stock Predictor (Multi-Company)", layout="centered")
st.title("📊 Multi-Company Stock Prediction (Upload CSV)")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your stock CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file with columns: date, symbol, open, high, low, close, volume")
    st.stop()

# Load Data
df = pd.read_csv(uploaded_file)

# Basic Cleanup
try:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
except:
    st.error("Couldn't parse 'date' column. Please check format.")
    st.stop()

# Rename columns
df.rename(columns={
    'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW',
    'close': 'CLOSE', 'volume': 'VOLUME'
}, inplace=True)

required_cols = ['date', 'symbol', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
if not all(col in df.columns for col in required_cols):
    st.error("Required columns missing. Must have: date, symbol, open, high, low, close, volume")
    st.stop()

# Unique companies
symbols = df['symbol'].unique()
selected = st.selectbox("🔍 Select Company Symbol", symbols)

# Filter selected company
stock_df = df[df['symbol'] == selected].copy().sort_values('date')

# Add indicators
stock_df['SMA_14'] = TA.SMA(stock_df, 14)
stock_df['EMA_14'] = TA.EMA(stock_df, 14)
stock_df['RSI_14'] = TA.RSI(stock_df)
macd = TA.MACD(stock_df)
stock_df['MACD'] = macd['MACD']
stock_df['SIGNAL'] = macd['SIGNAL']

# Build feature dataset
model_df = stock_df[['CLOSE', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'SIGNAL']].dropna()
if len(model_df) < 50:
    st.warning("Not enough data after indicators. Try another company.")
    st.stop()

# Prepare sequences
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(model_df)
X, Y = [], []
n = 10
for i in range(n, len(scaled_data)):
    X.append(scaled_data[i-n:i])
    Y.append(scaled_data[i][0])
X, Y = np.array(X), np.array(Y)

# Train/test split
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

# Prediction
Y_pred = model.predict(X_test)
close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
Y_test_inv = close_scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_pred_inv = close_scaler.inverse_transform(Y_pred)

# Plotting
st.subheader(f"📈 {selected} - Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(Y_test_inv, label='Actual', color='blue')
ax.plot(Y_pred_inv, label='Predicted', color='red')
ax.legend()
st.pyplot(fig)
