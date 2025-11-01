import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
import pandas as pd
import numpy as np

class DinoMLPFusion(nn.Module):
    def __init__(self, num_classes, bio_dim):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("1aurent/vit_small_patch16_256.tcga_brca_dino")
        for param in self.backbone.parameters():
            param.requires_grad = False
        vit_out_dim = 384
        self.bio_proj = nn.Linear(bio_dim, 128)
        self.head = nn.Sequential(
            nn.Linear(vit_out_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, img, bio):
        with torch.no_grad():
            x_img = self.backbone(img).pooler_output
        x_bio = self.bio_proj(bio)
        x = torch.cat([x_img, x_bio], dim=1)
        return self.head(x)

# Biomarker encoding setup
BIOMARKERS = ['BRCA1', 'CDH1', 'EGFR', 'ERBB2', 'ESR1', 'Ki-67', 'MKI67', 'PGR', 'PTEN', 'RB1', 'SNAI', 'SNAI1', 'TP53']
INTENSITIES = ['Moderate', 'Negative', 'Not detected', 'Strong', 'Weak']
STAININGS = ['High', 'Low', 'Medium', 'Not detected']
SUBTYPES = ['IDC', 'ILC', 'MBC', 'TNBC']

def encode_biomarkers(biomarker, intensity, staining):
    """Encode biomarker information into one-hot vector"""
    # Create feature dict
    features = {}
    
    # Biomarker one-hot
    for bm in BIOMARKERS:
        features[f'biomarker_{bm}'] = 1 if bm == biomarker else 0
    
    # Intensity one-hot
    for intens in INTENSITIES:
        features[f'intensity_{intens}'] = 1 if intens == intensity else 0
    
    # Staining one-hot
    for stain in STAININGS:
        features[f'staining_{stain}'] = 1 if stain == staining else 0
    
    # Convert to tensor
    feature_vector = torch.tensor(list(features.values()), dtype=torch.float32)
    return feature_vector

def load_model(model_path='dino_model.pth', device='cuda', from_hub=False, repo_id=None):
    """
    Load trained model
    
    Args:
        model_path: Local path to model file
        device: 'cuda' or 'cpu'
        from_hub: If True, download from Hugging Face Hub
        repo_id: Hugging Face repository ID (e.g., 'username/breast-cancer-dino')
    """
    # Download from Hugging Face if requested
    if from_hub:
        if repo_id is None:
            raise ValueError("repo_id must be provided when from_hub=True")
        print(f"Downloading model from Hugging Face: {repo_id}")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=repo_id, filename="dino_model.pth")
        print(f"Model downloaded to: {model_path}")
    
    # Calculate bio_dim based on encoding
    bio_dim = len(BIOMARKERS) + len(INTENSITIES) + len(STAININGS)
    num_classes = len(SUBTYPES)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = DinoMLPFusion(num_classes, bio_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

def preprocess_image(image_path):
    """Preprocess image for model input"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict_image(image_path, biomarker, intensity, staining, model_path='dino_model.pth'):
    """
    Predict breast cancer subtype from image and biomarker data
    
    Args:
        image_path: Path to histopathology image
        biomarker: Biomarker name (e.g., 'Ki-67')
        intensity: Intensity level (e.g., 'Strong', 'Moderate')
        staining: Staining pattern (e.g., 'High', 'Medium')
        model_path: Path to trained model weights
    
    Returns:
        dict with prediction results
    """
    # Load model
    model, device = load_model(model_path)
    
    # Preprocess inputs
    img_tensor = preprocess_image(image_path).to(device)
    bio_tensor = encode_biomarkers(biomarker, intensity, staining).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor, bio_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Get all probabilities
    all_probs = {SUBTYPES[i]: probabilities[0, i].item() for i in range(len(SUBTYPES))}
    
    return {
        'subtype': SUBTYPES[predicted_class],
        'confidence': confidence,
        'all_probabilities': all_probs
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Breast Cancer Subtype Classification')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--biomarker', type=str, required=True, help='Biomarker name')
    parser.add_argument('--intensity', type=str, required=True, help='Intensity level')
    parser.add_argument('--staining', type=str, required=True, help='Staining pattern')
    parser.add_argument('--model', type=str, default='dino_model.pth', help='Model path')
    
    args = parser.parse_args()
    
    result = predict_image(
        image_path=args.image,
        biomarker=args.biomarker,
        intensity=args.intensity,
        staining=args.staining,
        model_path=args.model
    )
    
    print(f"\nPrediction Results:")
    print(f"Predicted Subtype: {result['subtype']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nAll Probabilities:")
    for subtype, prob in result['all_probabilities'].items():
        print(f"  {subtype}: {prob:.4f}")