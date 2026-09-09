from pathlib import Path

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

dataset = datasets.CIFAR10(
	root=Path("C:/data"),
	train=True,
	download=False,
	transform=transforms.ToTensor(),
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))
print("image shape", images.shape)
print("label shape", labels.shape)
print("label", labels[:5])

import matplotlib.pyplot as plt
img1 = images[0].permute(1,2,0)
plt.imshow(img1)
plt.show()