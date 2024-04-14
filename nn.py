from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

class MyModule(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    pass

class SimpleClassifier(nn.Module):
  def __init__(self, num_inputs, num_hidden, num_outputs):
    super().__init__()
    self.linear1 = nn.Linear(num_inputs, num_hidden)
    self.act_fn = nn.Tanh()
    self.linear2 = nn.Linear(num_hidden, num_outputs)
  
  def forward(self, x):
    x = self.linear1(x)
    x = self.act_fn(x)
    x = self.linear2(x)
    return x
    
model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
print(model)

for name, param in model.named_parameters():
  print(f"Parameter {name}, shape: {param.shape}")

import torch.utils.data as data

class XORDataset(data.Dataset):

  def __init__(self, size, std=0.1):
    super().__init__()
    self.size = size
    self.std = std
    self.generate_continuous_xor()

  def generate_continuous_xor(self):
    data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
    label = (data.sum(dim=1) == 1).to(torch.long)
    data += self.std * torch.randn(data.shape)

    self.data = data
    self.label = label

  def __len__(self):
    return self.size
  
  def __getitem__(self, idx):
    data_point = self.data[idx]
    data_label = self.label[idx]
    return data_point, data_label
  
dataset = XORDataset(size=200)
print("Size of dataset:", len(dataset))
print("Data point 0:", dataset[0])

def visualize_samples(data, label):
  if isinstance(data, torch.Tensor):
    data = data.cpu().numpy()
  if isinstance(label, torch.Tensor):
    label = label.cpu().numpy()
  data_0 = data[label == 0]
  data_1 = data[label == 1]

  plt.figure(figsize=(4,4))
  plt.scatter(data_0[:,0], data_0[:,1], edgecolors="#333", label="Class 0")
  plt.scatter(data_1[:,0], data_1[:,1], edgecolors="#333", label="Class 1")
  plt.title("Dataset samples")
  plt.ylabel(r"$x_2$")
  plt.xlabel(r"$x_1$")
  plt.legend()


# visualize_samples(dataset.data, dataset.label)
# plt.show()

data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
data_inputs, data_labels = next(iter(data_loader))
print("Data inputs", data_inputs.shape, "\n", data_inputs)
print("Data labels", data_labels.shape, "\n", data_labels)

loss_module = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_dataset = XORDataset(size=2500)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

model.to(device)

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
  model.train()
  for epoch in tqdm(range(num_epochs)):
    for data_inputs, data_labels in data_loader:
      data_inputs = data_inputs.to(device)
      data_labels = data_labels.to(device)
      preds = model(data_inputs)
      preds = preds.squeeze(dim=1)
      loss = loss_module(preds, data_labels.float())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

train_model(model, optimizer, train_data_loader, loss_module)
state_dict = model.state_dict()
print(state_dict)

torch.save(state_dict, "our_model.tar")

state_dict = torch.load("our_model.tar")
new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
new_model.load_state_dict(state_dict)

print("Original model\n", model.state_dict())
print("\nLoaded model\n", new_model.state_dict())

test_dataset = XORDataset(size=500)
test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=False)

def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

eval_model(model, test_data_loader)