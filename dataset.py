from pathlib import Path
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
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
        transform=train_transform,
    )
    
    test_dataset = datasets.CIFAR10(
        root=Path("C:/data"),
        train=False,
        download=False,
        transform=test_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Test dataloaders when running dataset.py directly
    train_loader, test_loader = get_dataloaders()
    train_image, train_labels = next(iter(train_loader))
    print(f"Train Batch Shape: {train_image.shape}")
    print(f"Train Batch Min: {train_image.min():.4f}, Max: {train_image.max():.4f}")