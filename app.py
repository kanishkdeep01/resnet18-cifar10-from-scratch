import torch
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
from model_resnet import ResNet  


device = torch.device("cpu") 
model = ResNet().to(device)
model.load_state_dict(torch.load("elevated_resnet_cifar10.pth", map_location=device))
model.eval() 


labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


inference_transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
])

def predict(img):
    
    tensor = inference_transform(img).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    
    return {labels[i]: float(probabilities[i]) for i in range(10)}


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3), 
    title="Elevated ResNet-18 Vision Model",
    description="Upload a photo of a plane, car, bird, cat, deer, dog, frog, horse, ship, or truck. Powered by a custom PyTorch architecture hitting 93.00% accuracy on CIFAR-10."
)

if __name__ == "__main__":
    interface.launch()