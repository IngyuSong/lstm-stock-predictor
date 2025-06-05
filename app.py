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

# LSTM 모델 클래스 정의
class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 시퀀스 생성 함수
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 세션 상태 초기화
if 'recent_tickers' not in st.session_state:
    st.session_state.recent_tickers = []
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

# 종목 입력
ticker = st.text_input("종목 코드를 입력하세요 (예: AAPL):")

# 예측 시작 버튼
predict_button = st.button("예측 시작")

# 엔터키 처리
if ticker and (predict_button or st.session_state.current_ticker != ticker):
    st.session_state.current_ticker = ticker
    
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
    
    # PyTorch 텐서로 변환
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    
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
    
    # 데이터 길이 맞추기
    test_size = len(predicted_values)
    actual_values = actual_values[-test_size:]
    dates = df.index[-test_size:]
    
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
        exchange_rate = float(yf.download("KRW=X", period="1d")['Close'].iloc[-1])
    except:
        exchange_rate = 1300.0  # 기본값 설정
    
    # 예측값 비교 및 향후 예측 정보
    st.subheader("예측 결과 분석")
    
    # 오늘의 예측값
    today_pred = float(predicted_values[-1])  # numpy 배열로 변환
    yesterday_actual = float(actual_values[-2])  # numpy 배열로 변환
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
    last_actual = float(actual_values[-1])  # numpy 배열로 변환
    last_pred = float(predicted_values[-1])  # numpy 배열로 변환
    
    for days_ahead in days:
        # 단순 선형 추세 계산 (최근 30일 기준)
        recent_trend = (last_pred - float(predicted_values[-30])) / 30
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
            if st.button(f"{recent_ticker} ×", key=f"recent_{recent_ticker}"):
                st.session_state.current_ticker = recent_ticker
                ticker = recent_ticker
                st.rerun()