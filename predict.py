import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

# 予測したい日時
predict_date = pd.to_datetime("2023-04-01 12:00:00", utc=True)

# 日時情報をとく微量に変換
features = pd.DataFrame({
  "year": [predict_date.year],
  "month": [predict_date.month],
  "day": [predict_date.day],
  "hour": [predict_date.hour],
})

scaler = MinMaxScaler()
features_scaled = scaler.transform(features)

# とく微量をテンソルに変換
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

# モデルを評価モードに設定
model = torch.load("model.pth")
model.eval()

with torch.no_grad():
  predicted_price = model(features_tensor).item()

print(f'Predicted Price: {predicted_price}')
