from torch import nn
import torch

# ニューラルネットワークモデルを定義する
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 10)  # 入力層→隠れ層
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 2)  # 隠れ層→出力層

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x