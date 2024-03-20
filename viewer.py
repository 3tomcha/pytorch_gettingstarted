import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# MNISTデータセットをダウンロードし、Tensor形式に変換
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

# データローダーを作成
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# データローダーから一つのバッチを取得
images, labels = next(iter(test_loader))

# 最初の画像を表示
plt.imshow(images[0].squeeze(), cmap='gray')
plt.title(f'Label: {labels[0]}')
plt.show()
