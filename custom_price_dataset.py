import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import numpy as np
import joblib

data = pd.read_csv("./eth_btc_pricedata.csv")

# 無限大の値をNANに置き換える
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# NaN値を削除する
data.dropna(inplace=True)
# print(data.head())

# eth_timeを日時型に変換
# data["eth_time"] = pd.to_datetime(data["eth_time"], errors="coerce", utc=True)
data["btc_time"] = pd.to_datetime(data["btc_time"], errors="coerce", utc=True)


# 日時情報を数値特徴に変換（例：年、月、日、時間）
data["year"] = data["btc_time"].dt.year
data["month"] = data["btc_time"].dt.month
data["day"] = data["btc_time"].dt.day
data["hour"] = data["btc_time"].dt.hour

data.drop("eth_time", axis=1, inplace=True)
data.drop("eth_price", axis=1, inplace=True)
data.drop("eth_high", axis=1, inplace=True)
data.drop("eth_low", axis=1, inplace=True)
data.drop("eth_24hourhigh", axis=1, inplace=True)
data.drop("eth_24hourlow", axis=1, inplace=True)
data.drop("eth_nexthourprice", axis=1, inplace=True)
data.drop("eth_nextdayprice", axis=1, inplace=True)
data.drop("eth_nextweekprice", axis=1, inplace=True)

data.drop("btc_high", axis=1, inplace=True)
data.drop("btc_low", axis=1, inplace=True)
data.drop("btc_24hourhigh", axis=1, inplace=True)
data.drop("btc_24hourlow", axis=1, inplace=True)
data.drop("btc_nexthourprice", axis=1, inplace=True)
data.drop("btc_nextdayprice", axis=1, inplace=True)
data.drop("btc_nextweekprice", axis=1, inplace=True)


# 特微量とターゲットを定義する
X = data.drop(["btc_time", "btc_price"], axis=1)
y = data[["btc_price"]]

print(X.head())

# データの正規化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# scalerの保存
joblib.dump(scaler, "scaler.joblib")

# Numpy配列をPyTorchテンソルに変換する
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

# トレーニングセットとテストセットにデータを分割する
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# ニューラルネットワークモデルを定義する
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# モデルの初期化と損失関数、オプティマイザの定義
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 1
model = NeuralNetwork(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# モデルをトレーニングする
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test Loss: {test_loss.item():}.4f')

torch.save(model, "model.pth")