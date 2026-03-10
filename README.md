# Brain Tumor Detection using Deep Learning

This project is a deep learning based web application that detects brain tumors from MRI images.  
The system uses a fine-tuned ResNet-50 convolutional neural network to classify MRI scans into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

Users can upload an MRI image through an interactive Streamlit web interface, and the model predicts the tumor type along with a confidence score. The app also generates a Grad-CAM heatmap to highlight the regions of the MRI that influenced the model's prediction.

---

## Features

- Brain tumor classification using ResNet-50
- Upload MRI images for prediction
- Displays prediction with confidence score
- Grad-CAM heatmap for model explainability
- Interactive web interface built with Streamlit

---

## Model

The model is based on a **ResNet-50 architecture** and was trained on labeled brain MRI images.  
Transfer learning was used to fine-tune the network for multi-class tumor classification.

---

## Project Structure

Brain-Tumor-Detection
│
├── app.py # Streamlit application
├── final_brain_tumor_model.pth # Trained model weights
├── final_brain_tumor_model.ipynb # Model training notebook
├── requirements.txt # Python dependencies
├── runtime.txt # Python version for deployment
└── README.md # Project documentation

---

## Run the Streamlit app

https://brain-tumor-detection-ttnds7g4pxiydrhimzfans.streamlit.app/


---

## Usage

1. Upload a brain MRI image.
2. The model predicts the tumor type.
3. View the confidence score.
4. The Grad-CAM visualization highlights important regions used by the model.

---

## Technologies Used

- Python
- PyTorch
- ResNet-50
- Streamlit
- Grad-CAM
- NumPy / Matplotlib

