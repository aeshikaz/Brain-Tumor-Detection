import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

MODEL_PATH = "final_brain_tumor_model.pth"

class_names = ['glioma','meningioma','notumor','pituitary']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Target layer for Grad-CAM
target_layers = [model.layer4[-1]]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload MRI Image")

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI", width=300)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs,1)

    st.success(f"Prediction: {class_names[pred.item()]}")
    st.metric("Confidence", f"{confidence.item()*100:.2f}%")

    # Prepare image for GradCAM
    img_np = np.array(image.resize((224,224))).astype("float32") / 255.0

    # Grad-CAM
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    st.subheader("Tumor Localization (Grad-CAM)")

    st.image(visualization, caption="Model attention heatmap", width=300)