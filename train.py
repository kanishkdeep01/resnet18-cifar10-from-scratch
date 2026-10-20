import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_dataloader, test_dataloader
from model_resnet import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        total_correct += (predicted == labels).sum().item()
    train_accuracy = 100 * total_correct / total

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    
    test_accuracy = 100 * test_correct / test_total
    avg_loss = running_loss / len(train_dataloader)
    
    
    print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")   
        

torch.save(model.state_dict(), "elevated_resnet_cifar10.pth")
print("Model saved successfully as elevated_resnet_cifar10.pth!")
