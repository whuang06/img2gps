import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import math
from datasets import load_dataset
from model import ResnetClassifier
from model import ViTClassifier

BINS = 3
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/ec2-user/image2gps/scripts/5_4_3bins_model_randomresize_ratio0.75_scale0.7_finetuned.pth"
DATASET_PATH = "Willh96/image2gps_compressed_dataset"
MODEL_TYPE = "resnet"

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_bin_center(label, min_lat, max_lat, min_lon, max_lon, bins):
    row = label % bins
    col = label // bins
    
    lat_step = (max_lat - min_lat) / bins
    lon_step = (max_lon - min_lon) / bins
    
    center_lat = min_lat + (row + 0.5) * lat_step
    center_lon = min_lon + (col + 0.5) * lon_step
    return center_lat, center_lon

def get_label(lat, lon, min_lat, max_lat, min_lon, max_lon, bins):
    row = int(((lat - min_lat) / (max_lat - min_lat)) * bins)
    col = int(((lon - min_lon) / (max_lon - min_lon)) * bins)
    row = max(0, min(row, bins - 1))
    col = max(0, min(col, bins - 1))
    return row + (col * bins)

def get_bounds(dataset_dict):
    lats, lons = [], []
    for split in dataset_dict.keys():
        lats.extend(dataset_dict[split]["latitude"])
        lons.extend(dataset_dict[split]["longitude"])
    return min(lats), max(lats), min(lons), max(lons)

def evaluate_model():
    ds = load_dataset(DATASET_PATH)
    test_split = ds["test"]
    min_lat, max_lat, min_lon, max_lon = get_bounds(ds)

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def preprocess(examples):
        labels = [get_label(lat, lon, min_lat, max_lat, min_lon, max_lon, BINS) 
                  for lat, lon in zip(examples["latitude"], examples["longitude"])]
        pixel_values = [img_transforms(image.convert("RGB")) for image in examples["image"]]
        return {"pixel_values": pixel_values, "label": labels, "lat": examples["latitude"], "lon": examples["longitude"]}

    test_split.set_transform(preprocess)

    def collate_fn(examples):
        return {
            "images": torch.stack([ex["pixel_values"] for ex in examples]),
            "labels": torch.tensor([ex["label"] for ex in examples]),
            "true_coords": [(ex["lat"], ex["lon"]) for ex in examples]
        }

    test_loader = DataLoader(test_split, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    if MODEL_TYPE == "resnet":
        model = ResnetClassifier(num_classes=BINS * BINS)
    elif MODEL_TYPE == "vit":
        model = ViTClassifier(num_classes=BINS * BINS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0
    distances = []

    print(f"Starting test on {len(test_split)} images...")
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            true_coords = batch["true_coords"]

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for i in range(len(preds)):
                pred_label = preds[i].item()
                p_lat, p_lon = get_bin_center(pred_label, min_lat, max_lat, min_lon, max_lon, BINS)
                t_lat, t_lon = true_coords[i]
                
                dist = haversine_distance(t_lat, t_lon, p_lat, p_lon)
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
    evaluate_model()