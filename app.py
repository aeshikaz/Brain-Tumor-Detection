import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "final_brain_tumor_model.pth"

class_names = ['glioma','meningioma','notumor','pituitary']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

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

    st.image(image)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs,1)

    st.write("Prediction:", class_names[pred.item()])
    st.write("Confidence:", float(confidence))