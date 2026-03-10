import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
import types

sys.modules["cv2"] = types.ModuleType("cv2")
sys.modules["cv2"].COLORMAP_JET = None

from pytorch_grad_cam import GradCAM

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

MODEL_PATH = "final_brain_tumor_model.pth"

class_names = ['glioma','meningioma','notumor','pituitary']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


model = load_model()

target_layers = [model.layer4[-1]]

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

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs,1)

    st.success(f"Prediction: {class_names[pred.item()]}")
    st.metric("Confidence", f"{confidence.item()*100:.2f}%")

    img_np = np.array(image.resize((224,224))).astype("float32") / 255.0

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    fig, ax = plt.subplots()

    ax.imshow(img_np)
    ax.imshow(grayscale_cam, cmap="jet", alpha=0.5)
    ax.axis("off")

    st.pyplot(fig)