import streamlit as st
import yfinance as yf
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel
import datetime
import torch.nn as nn
import torch.optim as optim

st.title("주식 가격 예측 LSTM 웹앱")

# 세션 상태 초기화
if 'recent_tickers' not in st.session_state:
    st.session_state.recent_tickers = []

# 종목 입력
ticker = st.text_input("종목 코드를 입력하세요 (예: AAPL):", key="ticker_input")

# 예측 시작 버튼
predict_button = st.button("예측 시작")

# 엔터키 처리
if ticker and (predict_button or st.session_state.get('ticker_input') != ticker):
    st.session_state['ticker_input'] = ticker
    
    # 최근 검색 종목에 추가 (중복 제거)
    if ticker not in st.session_state.recent_tickers:
        st.session_state.recent_tickers.insert(0, ticker)
        # 최대 5개까지만 유지
        st.session_state.recent_tickers = st.session_state.recent_tickers[:5]
    
    # 데이터 가져오기
    df = yf.download(ticker, start="2020-01-01", end=datetime.datetime.now().strftime("%Y-%m-%d"))
    
    if len(df) == 0:
        st.error("데이터를 가져올 수 없습니다. 종목 코드를 확인해주세요.")
        st.stop()
    
    # 데이터 전처리
    df = df[['Close']]
    df = df.dropna()
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # 시퀀스 생성
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    
    # 데이터 분할
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 모델 학습
    model = StockPredictor(input_dim=1, hidden_dim=50, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 학습 진행
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 진행률 업데이트
        progress = (epoch + 1) / 100
        progress_bar.progress(progress)
        status_text.text(f"학습 진행 중... {int(progress * 100)}%")
    
    progress_bar.empty()
    status_text.empty()
    
    # 예측
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
    
    # 환율 정보 가져오기
    try:
        exchange_rate = yf.download("KRW=X", period="1d")['Close'].iloc[-1]
    except:
        exchange_rate = 1300  # 기본값 설정
    
    # 예측값 비교 및 향후 예측 정보
    st.subheader("예측 결과 분석")
    
    # 오늘의 예측값
    today_pred = predicted_values[-1]
    yesterday_actual = actual_values[-2]
    change = ((today_pred - yesterday_actual) / yesterday_actual) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="오늘의 예측 주가 (USD)",
            value=f"${today_pred:.2f}",
            delta=f"{change:.2f}%"
        )
    with col2:
        st.metric(
            label="오늘의 예측 주가 (KRW)",
            value=f"₩{today_pred * exchange_rate:,.0f}",
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"{days_ahead}일 후 예상 주가 (USD)",
                value=f"${future_pred:.2f}",
                delta=f"{future_change:.2f}%"
            )
        with col2:
            st.metric(
                label=f"{days_ahead}일 후 예상 주가 (KRW)",
                value=f"₩{future_pred * exchange_rate:,.0f}",
                delta=f"{future_change:.2f}%"
            )
    
    # 차트 표시 (예측값만)
    st.subheader("주가 예측 차트")
    st.line_chart(chart_data["예측"])

# 최근 검색 종목 목록 표시
if st.session_state.recent_tickers:
    st.subheader("최근 검색 종목")
    cols = st.columns(5)
    for idx, recent_ticker in enumerate(st.session_state.recent_tickers):
        with cols[idx]:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(recent_ticker, key=f"recent_{recent_ticker}"):
                    st.session_state['ticker_input'] = recent_ticker
                    st.experimental_rerun()
            with col2:
                if st.button("×", key=f"delete_{recent_ticker}"):
                    st.session_state.recent_tickers.remove(recent_ticker)
                    st.experimental_rerun()