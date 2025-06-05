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
    
    # 데이터프레임 생성 전에 차원 확인 및 변환
    actual_values = df.Close[seq_length:].values.flatten()
    predicted_values = preds_rescaled.reshape(-1)
    
    # 인덱스 생성
    dates = df.index[seq_length:]
    
    # 데이터프레임 생성 (인덱스 명시적 지정)
    chart_data = pd.DataFrame(
        {
            "실제": actual_values,
            "예측": predicted_values
        },
        index=dates
    )
    
    st.line_chart(chart_data)
    
    # 예측값 비교 및 향후 예측 정보
    st.subheader("예측 결과 분석")
    
    # 오늘의 예측값
    today_pred = predicted_values[-1]
    yesterday_actual = actual_values[-2]
    change = ((today_pred - yesterday_actual) / yesterday_actual) * 100
    
    st.metric(
        label="오늘의 예측 주가",
        value=f"{today_pred:.2f}",
        delta=f"{change:.2f}%"
    )
    
    # 향후 예측 (단순 선형 외삽)
    days = [7, 30, 365]  # 일주일, 한달, 1년
    last_actual = actual_values[-1]
    last_pred = predicted_values[-1]
    
    for days_ahead in days:
        # 단순 선형 추세 계산 (최근 30일 기준)
        recent_trend = (last_pred - predicted_values[-30]) / 30
        future_pred = last_pred + (recent_trend * days_ahead)
        future_change = ((future_pred - last_actual) / last_actual) * 100
        
        st.metric(
            label=f"{days_ahead}일 후 예상 주가",
            value=f"{future_pred:.2f}",
            delta=f"{future_change:.2f}%"
        )