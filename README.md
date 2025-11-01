# Breast Cancer Subtype Classification using ViT-DINO and Biomarkers

A deep learning project for classifying breast cancer subtypes (TNBC, IDC, MBC, ILC) using histopathology images and biomarker data.

## Overview

This project implements a multimodal fusion approach combining:
- Vision Transformer (ViT-DINO) pretrained on TCGA-BRCA dataset for image feature extraction
- Biomarker data (biomarker type, intensity, staining) processed through a projection network
- MLP fusion head for final classification

## Model Architecture

The model consists of three main components:

1. **Image Encoder**: `1aurent/vit_small_patch16_256.tcga_brca_dino` (frozen backbone)
2. **Biomarker Encoder**: Linear projection layer (128 dimensions)
3. **Fusion Head**: 2-layer MLP with ReLU and Dropout (0.3)

## Results

Best validation accuracy: **67.24%** (Epoch 15)

| Metric | Value |
|--------|-------|
| Val Accuracy | 67.24% |
| Val Loss | 0.8478 |
| Train Accuracy | 71.69% |
| Train Loss | 0.7337 |

## Dataset

The dataset contains:
- 600 histopathology images from 4 breast cancer subtypes
- Biomarker information: type, intensity, and staining patterns
- Train/Test split: ~480/120 samples

### Subtypes
- **TNBC**: Triple-Negative Breast Cancer (151 samples)
- **IDC**: Invasive Ductal Carcinoma (150 samples)
- **MBC**: Medullary Breast Cancer (150 samples)
- **ILC**: Invasive Lobular Carcinoma (149 samples)

## Installation

```bash
git clone https://github.com/AnjaliiD/Breast-Cancer-Subtype-Classification.git
cd Breast-Cancer-Subtype-Classification
pip install -r requirements.txt
python predict.py --image path/to/image.jpg --biomarker Ki-67 --intensity Strong --staining High
```

## Usage

### Training

```python
# See the Jupyter notebook for full training pipeline
jupyter notebook notebook.ipynb
```

### Inference

```python
from predict import predict_image

# Predict single image
result = predict_image(
    image_path="path/to/image.jpg",
    biomarker="Ki-67",
    intensity="Strong",
    staining="High"
)
print(f"Predicted subtype: {result['subtype']}")
```

## Model Weights

Download from Hugging Face:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="AnjaliiD/breast-cancer-dino",
    filename="dino_model.pth"
)
```

Or direct download: https://huggingface.co/AnjaliiD/breast-cancer-dino/resolve/main/dino_model.pth

## Requirements

- Python 3.11+
- PyTorch 2.0+
- transformers
- torchvision
- pandas
- numpy
- Pillow
- scikit-learn
- tqdm

## Project Structure

```
├── dino_model.pth            
├── notebook.ipynb  
├── upload_model.py          
├── README.md              
├── requirements.txt            
└── .gitignore                  
```

## Training Details

- **Optimizer**: AdamW (lr=1e-4)
- **Loss**: CrossEntropyLoss
- **Batch Size**: 16
- **Epochs**: 15
- **Image Size**: 224×224
- **Normalization**: ImageNet statistics

## Biomarkers Used

The model processes the following biomarkers:
- BRCA1, CDH1, EGFR, ERBB2, ESR1, Ki-67, MKI67, PGR, PTEN, RB1, SNAI, SNAI1, TP53

## Contact

For questions or collaboration: anjalidesai0111@gmail.com
