---
title: Stock Predictor App
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: streamlit_app.py
pinned: false
---


# 🚀 Stock Predictor App

This is a real-time **NYSE stock price prediction** web app built using **GRU + LSTM** deep learning models. The app uses technical indicators like RSI, MACD, EMA, and SMA for better forecasting.

---

## 🔴 Live Demo

👉 [Try it on Hugging Face](https://huggingface.co/spaces/KrishnaGupta989/stock-predictor-app)

---

## 📦 Features

- Predicts NYSE stock closing prices
- Real-time chart with Actual vs Predicted graph
- Uses technical indicators: RSI, MACD, EMA, SMA
- Built with Streamlit and TensorFlow
- Fully deployed using GitHub + Hugging Face CI/CD

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- finta (technical indicators)
- scikit-learn
- TensorFlow / Keras
- Matplotlib

---

## 📁 Project Structure

├── streamlit_app.py # Main Streamlit app
├── prices.zip # NYSE dataset (zipped)
├── requirements.txt # Python dependencies
├── README.md # Project overview
└── .github/
└── workflows/
└── hf-sync.yml # GitHub Action for deploy

yaml
Copy
Edit

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
📜 License
MIT License

🙋‍♂️ Author
Made with ❤️ by Krishna Gupta
