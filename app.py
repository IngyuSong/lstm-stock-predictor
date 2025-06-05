import streamlit as st
import yfinance as yf
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel

st.title("주식 가격 예측 LSTM 웹앱")

ticker = st.text_input("종목 코드 입력 (예: AAPL, MSFT):", "AAPL")
if st.button("예측 시작"):

    df = yf.download(ticker, period="5y")[['Close']]
    st.line_chart(df.Close)

    # 데이터 전처리
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    seq_length = 20

    def create_sequences(data):
        xs = []
        for i in range(len(data) - seq_length):
            xs.append(data[i:i+seq_length])
        return np.array(xs)

    X_test = create_sequences(data_scaled)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # 모델 불러오기
    model = LSTMModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    with torch.no_grad():
        preds = model(X_test).numpy()

    preds_rescaled = scaler.inverse_transform(preds)

    st.line_chart(pd.DataFrame({
        "실제": df.Close[seq_length:].values,
        "예측": preds_rescaled.reshape(-1)
    }))