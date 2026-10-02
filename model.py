import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels = 128, kernel_size=3,padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = 2,stride=2)

        self.fc1 = nn.Linear(in_features=128*4*4,out_features=256)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256,out_features=10)
        

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = SimpleCNN()
    output = model(torch.randn(1,3,32,32))
    print("Output shape:", output.shape)  # Expected shape: [1, 10]