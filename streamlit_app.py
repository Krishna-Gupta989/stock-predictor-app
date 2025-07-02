import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finta import TA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout

st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📊 Multi-Company Stock Prediction (Clean & Accurate)")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your stock CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file with: date, symbol, open, high, low, close, volume")
    st.stop()

# Read & clean
df = pd.read_csv(uploaded_file)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Standardize columns
df.rename(columns={
    'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW',
    'close': 'CLOSE', 'volume': 'VOLUME'
}, inplace=True)

required_cols = ['date', 'symbol', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
if not all(col in df.columns for col in required_cols):
    st.error("CSV must contain: date, symbol, open, high, low, close, volume")
    st.stop()

symbols = df['symbol'].unique()
selected = st.selectbox("📌 Select Company Symbol", symbols)

# Filter selected company
stock_df = df[df['symbol'] == selected].copy().sort_values('date')

# ⛔ Skip if data too small
if len(stock_df) < 150:
    st.warning(f"⚠️ {selected} has only {len(stock_df)} rows. Minimum 150 required for reliable prediction.")
    st.stop()

# Add indicators
try:
    stock_df['SMA_14'] = TA.SMA(stock_df, 14)
    stock_df['EMA_14'] = TA.EMA(stock_df, 14)
    stock_df['RSI_14'] = TA.RSI(stock_df)
    macd = TA.MACD(stock_df)
    stock_df['MACD'] = macd['MACD']
    stock_df['SIGNAL'] = macd['SIGNAL']
except Exception as e:
    st.error(f"❌ Error adding indicators: {e}")
    st.stop()

model_df = stock_df[['CLOSE', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'SIGNAL']].dropna()

# ⛔ Still too small after indicators?
if len(model_df) < 50:
    st.warning("❌ Not enough data after indicator calculation.")
    st.stop()

# Scale + Sequence
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(model_df)

X, Y = [], []
n = 10
for i in range(n, len(scaled_data)):
    X.append(scaled_data[i-n:i])
    Y.append(scaled_data[i][0])
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
model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=0)

# Predict
Y_pred = model.predict(X_test)
close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
Y_test_inv = close_scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_pred_inv = close_scaler.inverse_transform(Y_pred)

# Plot
st.subheader(f"📉 {selected} - Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(Y_test_inv, label='Actual', color='blue')
ax.plot(Y_pred_inv, label='Predicted', color='red')
ax.legend()
st.pyplot(fig)

# Metrics
r2 = r2_score(Y_test_inv, Y_pred_inv)
rmse = mean_squared_error(Y_test_inv, Y_pred_inv, squared=False)
st.success(f"📊 R² Score: {r2:.4f} | RMSE: {rmse:.2f}")
