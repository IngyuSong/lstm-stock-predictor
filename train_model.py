
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_data(symbol='AAPL', start='2015-01-01', end='2022-12-31'):
    df = yf.download(symbol, start=start, end=end)
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    seq_length = 50
    x, y = [], []
    for i in range(seq_length, len(scaled)):
        x.append(scaled[i-seq_length:i])
        y.append(scaled[i])

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

def train():
    x, y, scaler = load_data()
    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ model.pth 저장 완료")

if __name__ == "__main__":
    train()
