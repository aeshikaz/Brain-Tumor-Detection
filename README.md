# Multi-Class Brain Tumor Classification Using Fine-Tuned ResNet50

This project presents a deep learning-based system for automated multi-class brain tumor classification using MRI scans from the BRISC 2025 dataset.

The model uses a fine-tuned ResNet50 backbone with transfer learning, Grad-CAM explainability, and an ablation-validated training pipeline to classify:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The project also includes a systematic ablation study evaluating:
- CLAHE preprocessing
- Single-stage vs two-stage fine tuning
- Weighted sampling strategies

---

# Best Performing Configuration

## ResNet50 + Single-Stage Fine-Tuning + No CLAHE

### Final Results
- Accuracy: **97.40%**
- AUC: **99.86%**
- Macro F1 Score: **97.48%**

Key finding:
> CLAHE preprocessing reduced performance on the standardized BRISC 2025 dataset, while single-stage fine-tuning produced the best results.

---

# Dataset

## BRISC 2025
- 6,000 contrast-enhanced T1-weighted MRI scans
- Expert radiologist annotated
- 4 classification categories
- Multi-plane MRI images

Classes:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

# Model Architecture

## Backbone
- ResNet50 pretrained on ImageNet

## Custom Classification Head

```python
Linear(2048 → 512)
ReLU
Dropout(0.4)
Linear(512 → 4)
Softmax
```

---

# Training Pipeline

## Preprocessing
- Resize to 224×224
- ImageNet normalization
- Data augmentation

## Data Augmentation
- Random horizontal flip
- Random rotation
- Color jitter

## Optimization
- AdamW optimizer
- Learning rate: 1e-5
- Cosine Annealing LR Scheduler
- Early stopping

## Class Imbalance Handling
- Weighted Random Sampler
- Weighted CrossEntropy Loss

---

# Explainable AI

Grad-CAM visualization is integrated to highlight MRI regions influencing predictions.

This improves:
- model interpretability
- clinical trustworthiness
- understanding of tumor localization

---

# Ablation Study

| Configuration | Accuracy |
|---|---|
| CLAHE + Two-Stage | 92.70% |
| No CLAHE + Two-Stage | 95.10% |
| No Weighted Sampling | 95.00% |
| No CLAHE + Single-Stage | **97.40%** |

---

# Features

- Brain tumor classification using ResNet50
- MRI image upload and prediction
- Confidence score display
- Grad-CAM heatmap generation
- Streamlit web application
- Ablation study analysis
- Explainable AI integration

---

# Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Streamlit
- Grad-CAM
- Google Colab

---

# Research Contributions

- First ablation study on BRISC 2025
- Demonstrated CLAHE can reduce performance on curated datasets
- Grad-CAM explainability analysis for all tumor classes
- Comparative ResNet50 vs DenseNet121 evaluation

---

# Repository Structure

```bash
├── app.py
├── README.md
├── requirements.txt
├── ResNet50_+_CLAHE_+_Two_Stage.ipynb
├── densenet121-single-stage.ipynb
├── confusion_matrix.png
├── gradcam_heatmap.png
└── runtime.txt
```

---

# Author

Aeshika Singh
