import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import os
from typing import Tuple

def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(path)
    
    img_col = next(c for c in df.columns if c.lower() in ['image_path', 'filepath', 'image', 'path', 'file_name'])
    lat_col = next(c for c in df.columns if c.lower() in ['latitude', 'lat'])
    lon_col = next(c for c in df.columns if c.lower() in ['longitude', 'lon'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    X_list = []
    y_list = []
    
    base_dir = os.path.dirname(path)
    
    for _, row in df.iterrows():
        img_full_path = os.path.join(base_dir, row[img_col])
        try:
            img = Image.open(img_full_path).convert("RGB")
            X_list.append(transform(img))
            y_list.append([row[lat_col], row[lon_col]])
        except Exception as e:
            print(f"Error loading image {img_full_path}: {e}")
            continue

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)
    
    return X, y