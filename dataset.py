from pathlib import Path
import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_transform = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = datasets.CIFAR10(
	root=Path("C:/data"),
	train=True,
	download=False,
	transform= train_transform,
)
test_dataset = datasets.CIFAR10(
	root = Path("C:/data"),
	train = False,
	download = False,
	transform = test_transform,
)

train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)

train_image,train_labels = next(iter(train_dataloader))
# Expected shape: [N, C, H, W] -> [32, 3, 32, 32]
print("Pixels:", train_image[0][0][0][:5])

img = train_image[0]
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

img_unnorm = (img*std)+mean
img_hwc = img_unnorm.permute(1,2,0)
#plt.imshow(img_hwc)
#plt.show()
# Fix line 35: grab tensor directly from the batch
img = train_image[0]

print(f"Train Batch Min: {train_image.min():.4f}, Max: {train_image.max():.4f}")
test_image, test_labels = next(iter(test_dataloader))
print(f"Test Batch Shape: {test_image.shape}")
print(f"Test Batch Min: {test_image.min():.4f}, Max: {test_image.max():.4f}")






