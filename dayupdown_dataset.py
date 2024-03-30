import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("BTC-2017min.csv")
df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
features = df[["open", "high", "low", "close"]]
target = df["Target"][:-1]

X_train, X_test, y_train, y_test = train_test_split(features[:-1], target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_dataset = TensorDataset(torch.Tensor(X_train_scaled), torch.Tensor(y_train.values).long()) 
test_dataset = TensorDataset(torch.Tensor(X_test_scaled), torch.Tensor(y_test.values).long()) 

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# ニューラルネットワークモデルの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # 入力層から隠れ層へ
        self.fc2 = nn.Linear(64, 32)  # 隠れ層
        self.fc3 = nn.Linear(32, 2)  # 出力層（上昇、下降の2クラス）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = Net()
crientation = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = crientation(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# test
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs. labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
