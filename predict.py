import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib
from torch import nn
import numpy as np

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
    
# 予測したい日時
predict_date = pd.to_datetime("2025-04-01 12:00:00", utc=True)

# 日時情報をとく微量に変換
features = pd.DataFrame({
  "year": [predict_date.year],
  "month": [predict_date.month],
  "day": [predict_date.day],
  "hour": [predict_date.hour],
})

scaler = joblib.load("scaler.joblib")
features_scaled = scaler.transform(features)

# とく微量をテンソルに変換
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

# モデルを評価モードに設定
model = torch.load("model.pth")
model.eval()

with torch.no_grad():
  predicted_price = model(features_tensor).item()

print(f'Predicted Price: {predicted_price}')

# 予測値のスケーリングを逆に戻すためのダミーの特徴量セットを作成
predicted_price_scaled = np.array([[predicted_price, 0, 0, 0]])  # 予測値を最初に、他はダミーの値

# 逆スケーリングを実行
predicted_price_real = scaler.inverse_transform(predicted_price_scaled)[0][0]  # 最初の特徴量の値を取得

print(f'Real Predicted Price: {predicted_price_real}')