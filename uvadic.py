# import os
# import math
import numpy as np
# import time

# import matplotlib.pyplot as plt
# from Ipython.display import set_matplotlib_formats
# set_matplotlib_formats("svg", "pdf")
# from matplotlib.colors import to_rgba
# import seaborn as sns
# sns.set()

# from tqdm.network import tpdm

import torch
print("using_torch", torch.__version__)
torch.manual_seed(42)
x = torch.Tensor(2, 3, 4)
print(x)
y = torch.Tensor([[1, 2], [3, 4]])
print(y)
z = torch.rand(2, 3, 4)
print(z)

shape = z.shape
print("shape: ", shape)

size = z.size()
print("size:", size)

dim1, dim2, dim3 = z.size()
print("size:", dim1, dim2, dim3)

np_arr = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(np_arr)

print("Numpy array:", np_arr)
print("PyTorch tensor:", tensor)

tensor = torch.arange(4)
np_arr = tensor.numpy()

print("PyTorch tensor:", tensor)
print("Numpy array:", np_arr)

x1 = torch.rand(2, 3)
x2 = torch.rand(2, 3)
y = x1 + x2

print("X1", x1)
print("X2", x2)
print("Y", y)

x1 = torch.rand(2, 3)
x2 = torch.rand(2, 3)
print("X1 (before)", x1)
print("X2 (before)", x2)

x2.add_(x1)
print("X1 (after)", x1)
print("X2 (after)", x2)

x = torch.arange(6)
print("X", x)

x = x.view(2, 3)
print("X", x)

x = x.permute(1, 0)
print("X", x)

x = torch.arange(6)
x = x.view(2, 3)
print("X", x)

W = torch.arange(9).view(3, 3)
print("W", W)

h = torch.matmul(x, W)
print("h", h)

x = torch.arange(12).view(3, 4)
print("X", x)

print(x[:, 1])
print(x[0])
print(x[:2, -1])
print(x[1:3, :])