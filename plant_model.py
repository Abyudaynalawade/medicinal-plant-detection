
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import warnings

warnings.filterwarnings("ignore")

# Class names
class_names = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avacado', 'Bamboo',
    'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka',
    'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon',
    'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Pappaya',
    'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel'
]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lazy-load model
_model = None
def load_model():
    global _model
    if _model is None:
        # Use a smaller model for Railway free tier
        _model = models.resnet18(weights=None)
        _model.fc = torch.nn.Sequential(
            torch.nn.Linear(_model.fc.in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, len(class_names))
        )
        # Load weights from relative path in repo (Linux compatible)
        weight_path = os.path.join(os.path.dirname(__file__), "resnet50_medicinal_plants_best.pth")
        _model.load_state_dict(torch.load(weight_path, map_location=device))
        _model.to(device)
        _model.eval()
    return _model

# Predict
def predict(image_path):
    model = load_model()
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_class_idx = torch.max(probs, 1)
        pred_class = class_names[pred_class_idx.item()]

    return pred_class, confidence.item()
