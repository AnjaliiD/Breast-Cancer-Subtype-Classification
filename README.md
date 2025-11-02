# Breast Cancer Subtype Classification using ViT-DINO

A multimodal deep learning model for classifying breast cancer subtypes using histopathology images and biomarker data.

## Overview

This project implements a fusion approach combining:
- Vision Transformer (ViT-DINO) pretrained on TCGA-BRCA dataset for image feature extraction
- Biomarker encoder for processing molecular markers (type, intensity, staining)
- MLP fusion head for final classification into 4 breast cancer subtypes

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 67.24% |
| Training Accuracy | 71.69% |
| Dataset Size | 600 images |
| Classes | 4 subtypes |

### Breast Cancer Subtypes
- **TNBC**: Triple-Negative Breast Cancer
- **IDC**: Invasive Ductal Carcinoma  
- **MBC**: Medullary Breast Cancer
- **ILC**: Invasive Lobular Carcinoma

## Quick Start

### Installation

```bash
git clone https://github.com/AnjaliiD/Breast-Cancer-Subtype-Classification.git
cd Breast-Cancer-Subtype-Classification
pip install -r requirements.txt
```

### Download Model Weights

The trained model is hosted on Hugging Face Hub:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="AnjaliiD/breast-cancer-dino",
    filename="dino_model.pth"
)
```

Direct download: https://huggingface.co/AnjaliiD/breast-cancer-dino/resolve/main/dino_model.pth

### Usage

#### Training Notebook
```bash
jupyter notebook notebook.ipynb
```

#### Inference Script

```bash
python predict.py \
    --image path/to/image.jpg \
    --biomarker Ki-67 \
    --intensity Strong \
    --staining High
```

Or use in Python:

```python
from predict import predict_image

result = predict_image(
    image_path="path/to/image.jpg",
    biomarker="Ki-67",
    intensity="Strong",
    staining="High"
)

print(f"Predicted Subtype: {result['subtype']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Model Architecture

```
Input Image (224x224)
    |
ViT-DINO Encoder (frozen)
    |
Image Features (384-dim)
    |                    <- Biomarker Features (26-dim)
    +-------- Concatenate --------+
              |
         MLP Head (256-dim)
              |
         4 Classes (TNBC, IDC, MBC, ILC)
```

### Components
- **Backbone**: `1aurent/vit_small_patch16_256.tcga_brca_dino` (frozen)
- **Biomarker Projection**: Linear layer (26 → 128 dim)
- **Classifier**: 2-layer MLP with ReLU and Dropout (0.3)

## Dataset

**Source**: Breast Cancer Detection dataset from Kaggle

### Structure
- **Total Images**: 600 histopathology images
- **Image Format**: JPEG (resized to 224×224)
- **Metadata**: CSV with biomarker information

### Biomarkers (13 types)
BRCA1, CDH1, EGFR, ERBB2, ESR1, Ki-67, MKI67, PGR, PTEN, RB1, SNAI, SNAI1, TP53

### Class Distribution
| Subtype | Count |
|---------|-------|
| TNBC    | 151   |
| IDC     | 150   |
| MBC     | 150   |
| ILC     | 149   |

## Training Details

- **Framework**: PyTorch 2.0+
- **Optimizer**: AdamW (learning rate: 1e-4)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 16
- **Epochs**: 15
- **GPU**: NVIDIA Tesla T4
- **Training Time**: ~50 minutes per epoch

## Methodology

### Data Preprocessing
1. Remove null values in intensity/staining fields
2. Stratified train/test split by subtype
3. One-hot encoding for categorical biomarker features

### Image Preprocessing
1. Resize to 224×224 pixels
2. Normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. Convert to RGB format

### Model Training
1. Freeze ViT-DINO backbone weights
2. Train only fusion head and biomarker encoder
3. Monitor validation accuracy for best model selection

## Training Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 68.60%    | 0.7941     | 61.21%  | 0.9113   |
| 3     | 72.93%    | 0.7484     | 66.38%  | 0.8731   |
| 5     | 71.69%    | 0.7337     | 67.24%  | 0.8478   |

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- torch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- huggingface_hub
- numpy, pandas, scikit-learn, Pillow

## Project Structure

```
.
├── notebook.ipynb          # Training notebook
├── predict.py              # Inference script
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # Project Documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ViT-DINO pretrained model from [1aurent/vit_small_patch16_256.tcga_brca_dino](https://huggingface.co/1aurent/vit_small_patch16_256.tcga_brca_dino)
- Breast Cancer Detection dataset from Kaggle

## Contact

**Anjali D**
- Email: anjalidesai0111@gmail.com
- GitHub: [@AnjaliiD](https://github.com/AnjaliiD)
- Hugging Face: [@AnjaliiD](https://huggingface.co/AnjaliiD)

## Links

- **Model on Hugging Face**: https://huggingface.co/AnjaliiD/breast-cancer-dino
- **Training Notebook**: [notebook.ipynb](notebook.ipynb)
