import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
from huggingface_hub import hf_hub_download

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

# Biomarker encoding setup - cleaned up duplicates
BIOMARKERS = ['BRCA1', 'CDH1', 'EGFR', 'ERBB2', 'ESR1', 'Ki-67', 'MKI67', 'PGR', 'PTEN', 'RB1', 'SNAI', 'SNAI1', 'TP53']
INTENSITIES = ['Moderate', 'Negative', 'Not detected', 'Strong', 'Weak']
STAININGS = ['High', 'Low', 'Medium', 'Not detected']
SUBTYPES = ['IDC', 'ILC', 'MBC', 'TNBC']

def encode_biomarkers(biomarker, intensity, staining):
    """
    Encode biomarker information into one-hot vector
    
    Args:
        biomarker: Biomarker name (e.g., 'Ki-67', 'ESR1')
        intensity: Intensity level ('Strong', 'Moderate', 'Weak', 'Negative', 'Not detected')
        staining: Staining pattern ('High', 'Medium', 'Low', 'Not detected')
    
    Returns:
        torch.Tensor: One-hot encoded feature vector
    """
    features = {}
    
    # Biomarker one-hot encoding
    for bm in BIOMARKERS:
        features[f'biomarker_{bm}'] = 1 if bm == biomarker else 0
    
    # Intensity one-hot encoding
    for intens in INTENSITIES:
        features[f'intensity_{intens}'] = 1 if intens == intensity else 0
    
    # Staining one-hot encoding
    for stain in STAININGS:
        features[f'staining_{stain}'] = 1 if stain == staining else 0
    
    # Convert to tensor
    feature_vector = torch.tensor(list(features.values()), dtype=torch.float32)
    return feature_vector

def load_model(model_path=None, device='cuda', from_hub=True, repo_id="AnjaliiD/breast-cancer-dino"):
    """
    Load trained model from Hugging Face Hub or local path
    
    Args:
        model_path: Local path to model file (used if from_hub=False)
        device: Device to load model on ('cuda' or 'cpu')
        from_hub: If True, download from Hugging Face Hub
        repo_id: Hugging Face repository ID
    
    Returns:
        tuple: (model, device)
    """
    # Download from Hugging Face if requested
    if from_hub:
        print(f"Downloading model from Hugging Face: {repo_id}")
        model_path = hf_hub_download(repo_id=repo_id, filename="dino_model.pth")
        print(f"Model downloaded to: {model_path}")
    elif model_path is None:
        raise ValueError("model_path must be provided when from_hub=False")
    
    # Calculate bio_dim based on encoding dimensions
    bio_dim = len(BIOMARKERS) + len(INTENSITIES) + len(STAININGS)
    num_classes = len(SUBTYPES)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = DinoMLPFusion(num_classes, bio_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

def preprocess_image(image_path):
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to input image
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def predict_image(image_path, biomarker, intensity, staining, 
                  model_path=None, from_hub=True, repo_id="AnjaliiD/breast-cancer-dino"):
    """
    Predict breast cancer subtype from image and biomarker data
    
    Args:
        image_path: Path to histopathology image
        biomarker: Biomarker name (e.g., 'Ki-67', 'ESR1', 'TP53')
        intensity: Intensity level ('Strong', 'Moderate', 'Weak', 'Negative', 'Not detected')
        staining: Staining pattern ('High', 'Medium', 'Low', 'Not detected')
        model_path: Path to local model weights (used if from_hub=False)
        from_hub: If True, download model from Hugging Face Hub
        repo_id: Hugging Face repository ID
    
    Returns:
        dict: Prediction results with keys:
            - subtype: Predicted cancer subtype
            - confidence: Confidence score (0-1)
            - all_probabilities: Dictionary of all class probabilities
    
    Example:
        >>> result = predict_image(
        ...     'sample.jpg', 
        ...     'Ki-67', 
        ...     'Strong', 
        ...     'High'
        ... )
        >>> print(result['subtype'])
        'TNBC'
    """
    # Load model
    model, device = load_model(model_path, device='cuda', from_hub=from_hub, repo_id=repo_id)
    
    # Preprocess inputs
    img_tensor = preprocess_image(image_path).to(device)
    bio_tensor = encode_biomarkers(biomarker, intensity, staining).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor, bio_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Get all class probabilities
    all_probs = {SUBTYPES[i]: probabilities[0, i].item() for i in range(len(SUBTYPES))}
    
    return {
        'subtype': SUBTYPES[predicted_class],
        'confidence': confidence,
        'all_probabilities': all_probs
    }

def main():
    """Command-line interface for prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Breast Cancer Subtype Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image sample.jpg --biomarker Ki-67 --intensity Strong --staining High
  python predict.py --image sample.jpg --biomarker ESR1 --intensity Moderate --staining Medium --local --model dino_model.pth

Valid Values:
  Biomarkers: BRCA1, CDH1, EGFR, ERBB2, ESR1, Ki-67, MKI67, PGR, PTEN, RB1, SNAI, SNAI1, TP53
  Intensities: Strong, Moderate, Weak, Negative, Not detected
  Stainings: High, Medium, Low, Not detected
        """
    )
    
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to histopathology image')
    parser.add_argument('--biomarker', type=str, required=True, 
                        help='Biomarker name (e.g., Ki-67, ESR1)')
    parser.add_argument('--intensity', type=str, required=True, 
                        help='Intensity level (Strong, Moderate, Weak, Negative, Not detected)')
    parser.add_argument('--staining', type=str, required=True, 
                        help='Staining pattern (High, Medium, Low, Not detected)')
    parser.add_argument('--model', type=str, default=None, 
                        help='Local model path (optional, used with --local)')
    parser.add_argument('--local', action='store_true', 
                        help='Use local model instead of downloading from Hugging Face Hub')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.biomarker not in BIOMARKERS:
        print(f"Error: Invalid biomarker '{args.biomarker}'")
        print(f"Valid biomarkers: {', '.join(BIOMARKERS)}")
        return
    
    if args.intensity not in INTENSITIES:
        print(f"Error: Invalid intensity '{args.intensity}'")
        print(f"Valid intensities: {', '.join(INTENSITIES)}")
        return
    
    if args.staining not in STAININGS:
        print(f"Error: Invalid staining '{args.staining}'")
        print(f"Valid stainings: {', '.join(STAININGS)}")
        return
    
    # Make prediction
    try:
        result = predict_image(
            image_path=args.image,
            biomarker=args.biomarker,
            intensity=args.intensity,
            staining=args.staining,
            model_path=args.model,
            from_hub=not args.local
        )
        
        # Display results
        print("\n" + "="*50)
        print("Prediction Results")
        print("="*50)
        print(f"Predicted Subtype: {result['subtype']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll Class Probabilities:")
        for subtype, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 30)
            print(f"  {subtype:4s}: {prob:.2%} {bar}")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
