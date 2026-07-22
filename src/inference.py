# src/inference.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from models import get_benchmark_model

def predict_single_image(model_path, image_path, model_name="af_vit_magnet", device=None):
    """
    Loads pre-trained classification checkpoint and predicts class probabilities for a single image.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Benign", "Cancer", "Normal"]
    
    model = get_benchmark_model(model_name, num_classes=3, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # Image preparation
    img = Image.open(image_path).convert("L")
    img = img.resize((224, 224), Image.BILINEAR)
    img_np = np.stack([np.array(img)] * 3, axis=-1).astype(np.float32) / 255.0
    
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.25]*3)
    ])
    
    tensor = tf(Image.fromarray((img_np * 255).astype(np.uint8))).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    results = {
        "predicted_class": class_names[pred_idx],
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }
    return results