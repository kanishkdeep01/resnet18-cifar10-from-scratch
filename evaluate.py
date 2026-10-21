import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_dataloaders
from model_resnet import ResNet

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data & model
    _, test_loader = get_dataloaders(batch_size=64)
    model = ResNet().to(device)
    model.load_state_dict(torch.load("elevated_resnet_cifar10.pth", map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("Evaluating 10,000 test images...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    
    print("\n--- Per-Class Performance Report ---")
    print(classification_report(all_targets, all_preds, target_names=classes, digits=4))

   
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Elevated ResNet-18 - CIFAR-10 Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Saved confusion matrix heatmap to 'confusion_matrix.png'!")

if __name__ == "__main__":
    run_evaluation()