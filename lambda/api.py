from flask import Flask
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib
from torch import nn
import numpy as np
from flask import jsonify
from neural_network import NeuralNetwork
    
app = Flask(__name__)
CORS(app)

@app.route("/cats")
def cats():
  return "Cats"

@app.route("/dogs/<id>")
def dog(id):
  return "Dog"

@app.route("/predict")    
def predict():      
  # 予測したい日時
  predict_date = pd.to_datetime("2025-04-01 12:00:00", utc=True)

  # 日時情報をとく微量に変換
  features = pd.DataFrame({
    "year": [predict_date.year],
    "month": [predict_date.month],
    "day": [predict_date.day],
    "hour": [predict_date.hour],
  })

  scaler = joblib.load("scaler2.joblib")
  features_scaled = scaler.transform(features)

  # とく微量をテンソルに変換
  features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

  # モデルを評価モードに設定
  model = torch.load("model2.pth")
  model.eval()

  with torch.no_grad():
    predicted_price = model(features_tensor).item()

  print(f'Predicted Price: {predicted_price}')

  # 予測値のスケーリングを逆に戻すためのダミーの特徴量セットを作成
  predicted_price_scaled = np.array([[predicted_price, 0, 0, 0]])  # 予測値を最初に、他はダミーの値

  # 逆スケーリングを実行
  predicted_price_real = scaler.inverse_transform(predicted_price_scaled)[0][0]  # 最初の特徴量の値を取得

  print(f'Real Predicted Price: {predicted_price_real}')

  return jsonify({'Predicted Price': predicted_price_real})