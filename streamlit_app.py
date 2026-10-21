import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_resnet import ResNet

st.set_page_config(page_title="Elevated ResNet-18 Demo", page_icon="🖼️")

st.title("Elevated ResNet-18: CIFAR-10 Classifier")
st.write("Upload any image to test the from-scratch PyTorch ResNet-18 model (93.20% test accuracy).")

@st.cache_resource
def load_model():
    model = ResNet()
    model.load_state_dict(torch.load("elevated_resnet_cifar10.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Input", use_container_width=True)
        
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        
    top_prob, top_class = torch.max(probs, 0)
    
    with col2:
        st.markdown(f"### Prediction: **{classes[top_class.item()].upper()}**")
        st.markdown(f"**Confidence:** `{top_prob.item()*100:.2f}%`")
        
    st.write("---")
    st.write("### Class Probability Breakdown")
    prob_dict = {classes[i]: float(probs[i]) for i in range(10)}
    st.bar_chart(prob_dict)