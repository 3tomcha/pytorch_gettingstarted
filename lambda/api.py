from pkg_resources import require
from flask import Flask
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib
from torch import nn
import numpy as np
from flask import Flask, request, jsonify
from neural_network import SimpleNN
    
app = Flask(__name__)
CORS(app)

@app.route("/cats")
def cats():
  return "Cats"

@app.route("/dogs/<id>")
def dog(id):
  return "Dog"

model = SimpleNN()
model.load_state_dict(torch.load("1hupdown_model_state.pth"))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    input_data = torch.FloatTensor([data['features']])
    outputs = model(input_data)
    _, predicted = torch.max(outputs.data, 1)
    return jsonify({'prediction': predicted.item()})
