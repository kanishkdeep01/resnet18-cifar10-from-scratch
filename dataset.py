from pathlib import Path

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

my_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(
	root=Path("C:/data"),
	train=True,
	download=False,
	transform= my_transform,
)
test_dataset = datasets.CIFAR10(
	root = Path("C:/data"),
	train = False,
	download = False,
	transform = my_transform,
)

train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)

train_image,train_labels = next(iter(train_dataloader))

print("Pixels:", train_image[0][0][0][:5])





