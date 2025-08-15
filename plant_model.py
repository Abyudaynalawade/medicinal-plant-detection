import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import warnings
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# Class names
class_names = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avacado', 'Bamboo',
    'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka',
    'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon',
    'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Pappaya',
    'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel'
]

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)  # limit CPU usage

# Lazy-load model
_model = None
def load_model():
    global _model
    if _model is None:
        # Use ResNet50 since your weights are trained on it
        _model = models.resnet50(weights=None)
        _model.fc = torch.nn.Sequential(
            torch.nn.Linear(_model.fc.in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, len(class_names))
        )

        # Download the weights from Hugging Face Hub
        weight_path = hf_hub_download(
            repo_id="Abhyuday03/Medicinal_plant_detection",
            filename="resnet50_medicinal_plants_best.pth"
        )

        # Load weights
        _model.load_state_dict(torch.load(weight_path, map_location=device))
        _model.to(device)
        _model.eval()
    return _model

# Prediction function
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


if __name__ == "__main__":
    test_image = "path_to_your_test_image.jpg"  # change to your image path
    label, conf = predict(test_image)
    print(f"Prediction: {label} (Confidence: {conf:.2f})")
