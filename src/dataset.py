# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MammographyClassificationDataset(Dataset):
    """
    Dataset loader for clean mammography images mapped to BI-RADS-aligned categories.
    Expects directory structure: root/<class_name>/*.png
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Label mapping: 0 = Benign, 1 = Cancer, 2 = Normal
        self.label_map = {"Benign": 0, "Cancer": 1, "Normal": 2}
        self.classes = sorted(list(self.label_map.keys()))
        
        for cls_name, label in self.label_map.items():
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        
        # Replicate single grayscale channel to 3 channels for vision backbones
        img_np = np.array(image).astype(np.float32) / 255.0
        img_rgb = np.stack([img_np] * 3, axis=-1)
        image_pil = Image.fromarray((img_rgb * 255).astype(np.uint8))

        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float()

        return image_tensor, label, img_path