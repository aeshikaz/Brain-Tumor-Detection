# Brain Tumor Detection

This project is a deep learning web application that detects brain tumors from MRI images.  
Users can upload an MRI scan and the model predicts whether a tumor is present and its type.

The model is built using **ResNet50** and deployed as an interactive web app using **Streamlit**.

## Live Demo
https://brain-tumor-detection-ttnds7g4pxiydrhimzfans.streamlit.app

## How it works
1. Upload an MRI brain scan.
2. The image is preprocessed and passed through the trained ResNet50 model.
3. The system predicts the tumor class and shows a confidence score.

## Tech Stack
- Python  
- PyTorch  
- Streamlit  
- NumPy  
- OpenCV  
- Pillow  
