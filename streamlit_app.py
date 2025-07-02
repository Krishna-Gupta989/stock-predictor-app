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
st.title("📊 Smart Stock Predictor (Auto Column Mapping)")

uploaded_file = st.file_uploader("📤 Upload your stock CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file with any reasonable column names.")
    st.stop()

# Read file
df = pd.read_csv(uploaded_file)

# -----------------------
# ✅ Smart column matching
# -----------------------
column_map = {
    'date': ['date', 'timestamp', 'datetime'],
    'symbol': ['symbol', 'stock', 'ticker'],
    'open': ['open', 'open_price'],
    'high': ['high', 'high_price'],
    'low': ['low', 'low_price'],
    'close': ['close', 'closing', 'close_price', 'price'],
    'volume': ['volume', 'vol', 'traded']
}

matched = {}
for key, aliases in column_map.items():
    for col in df.columns:
        if str(col).strip().lower() in [a.lower() for a in aliases]:
            matched[key] = col
            break

# ✅ Check for required columns
required = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
missing = [col for col in required if col not in matched]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}")
    st.stop()

# ✅ Rename columns to standard names
df = df.rename(columns={v: k.upper() for k, v in matched.items()})
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['SYMBOL'] = df['SYMBOL'].astype(str).str.strip()
df = df.dropna(subset=['DATE'])

# ---------------------------
# 📌 Company selection dropdown
# ---------------------------
symbols = df['SYMBOL'].unique()
selected = st.selectbox("📌 Select Company Symbol", sorted(symbols))

stock_df = df[df['SYMBOL'] == selected].copy().sort_values('DATE')
if len(stock_df) < 150:
    st.warning(f"{selected} has only {len(stock_df)} rows. Minimum 150 required.")
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
    st.error(f"❌ Indicator error: {e}")
    st.stop()

model_df = stock_df[['CLOSE', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'SIGNAL']].dropna()
if len(model_df) < 50:
    st.warning("Not enough data after indicators.")
    st.stop()

# Sequence preparation
scaler = MinMaxScaler()
scaled = scaler.fit_transform(model_df)
X, Y = [], []
n = 10
for i in range(n, len(scaled)):
    X.append(scaled[i-n:i])
    Y.append(scaled[i][0])
X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

if len(X_train) == 0 or len(Y_train) == 0:
    st.warning("Not enough training data. Try another company.")
    st.stop()

# Model
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=0)

# Prediction
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
rmse = np.sqrt(mean_squared_error(Y_test_inv, Y_pred_inv))
st.success(f"📊 R² Score: {r2:.4f} | RMSE: {rmse:.2f}")
