import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import math
from model import ResnetClassifier
from model import ViTClassifier

class ExternalTestDataset(Dataset):
    def __init__(self, csv_path, parent_dir, transform=None):
        self.metadata = pd.read_csv(csv_path)
        self.parent_dir = parent_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_relative_path = row['image_file']
        lat = row['Latitude']
        lon = row['Longitude']

        img_path = os.path.join(self.parent_dir, img_relative_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Could not find {img_path}. Check your path joining logic.")
            raise

        if self.transform:
            image = self.transform(image)

        return image, lat, lon

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_label(lat, lon, min_lat, max_lat, min_lon, max_lon, bins):
    row = int(((lat - min_lat) / (max_lat - min_lat)) * bins)
    col = int(((lon - min_lon) / (max_lon - min_lon)) * bins)
    row = max(0, min(row, bins - 1))
    col = max(0, min(col, bins - 1))
    return row + (col * bins)

def get_bin_center(label, min_lat, max_lat, min_lon, max_lon, bins):
    row = label % bins
    col = label // bins
    lat_step = (max_lat - min_lat) / bins
    lon_step = (max_lon - min_lon) / bins
    return min_lat + (row + 0.5) * lat_step, min_lon + (col + 0.5) * lon_step

def evaluate_external_data():
    CSV_PATH = "/home/ec2-user/image2gps/playing_around_with_sample/sample_data/metadata.csv"
    PARENT_DIR = "/home/ec2-user/image2gps/playing_around_with_sample/sample_data" 
    MODEL_PATH = "/home/ec2-user/image2gps/scripts/5_4_3bins_model_randomresize_ratio0.75_scale0.7_finetuned.pth"
    MODEL_TYPE = "resnet"
    MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 39.950375, 39.95277777777778, -75.19258888888889, -75.17618611111112
    BINS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = ExternalTestDataset(CSV_PATH, PARENT_DIR, transform=img_transforms)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    if MODEL_TYPE == "resnet":
        model = ResnetClassifier(num_classes=BINS * BINS)
    elif MODEL_TYPE == "vit":
        model = ViTClassifier(num_classes=BINS * BINS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    distances = []
    correct = 0
    total = 0

    print(f"Testing on {len(test_ds)} external samples...")

    with torch.no_grad():
        for images, lats, lons in test_loader:
            images = images.to(DEVICE)
            
            labels = torch.tensor([
                get_label(lat.item(), lon.item(), MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, BINS)
                for lat, lon in zip(lats, lons)
            ]).to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for i in range(len(preds)):
                p_lat, p_lon = get_bin_center(preds[i].item(), MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, BINS)
                dist = haversine_distance(lats[i].item(), lons[i].item(), p_lat, p_lon)
                distances.append(dist)
    accuracy = 100 * correct / total
    median_error = np.median(distances)
    mean_error = np.mean(distances)
    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    print(f"Classification Accuracy: {accuracy:.2f}%")
    print(f"Mean Distance Error:     {mean_error:.2f} meters")
    print(f"Median Distance Error:   {median_error:.2f} meters")
    print("="*40)

if __name__ == "__main__":
    evaluate_external_data()