from flask import Flask
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib
from torch import nn
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/cats")
def cats():
  return "Cats"

@app.route("/dogs/<id>")
def dog(id):
  return "Dog"
