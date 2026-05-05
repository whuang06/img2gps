import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
import numpy as np
from model import ResnetClassifier
from model import ViTClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BINS = 3
DATASET_PATH = "Willh96/image2gps_compressed_dataset"
SPLIT = "train"
MODEL_PATH = "/home/ec2-user/image2gps/scripts/5_4_3bins_model_randomresize_ratio0.75_scale0.7_finetuned_hf.pth"
MODEL_TYPE = "resnet"

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        if MODEL_TYPE == "resnet":
            self.backbone = nn.Sequential(*list(original_model.resnet.children())[:-1])
            self._kind = "resnet"
        elif MODEL_TYPE == "vit":
            self.vit = original_model.vit
            self._kind = "vit"

    def forward(self, x):
        if self._kind == "resnet":
            x = self.backbone(x)
            return torch.flatten(x, 1)
        vit = self.vit
        x = vit._process_input(x)
        n = x.shape[0]
        cls = vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = vit.encoder(x)
        return x[:, 0]

def save_embeddings():
    if MODEL_TYPE == "resnet":
        base_model = ResnetClassifier(num_classes=BINS*BINS)
    elif MODEL_TYPE == "vit":
        base_model = ViTClassifier(num_classes=BINS*BINS)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = FeatureExtractor(base_model).to(DEVICE)
    model.eval()

    ds = load_dataset(DATASET_PATH)[SPLIT]

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def preprocess(examples):
        pixel_values = [img_transforms(img.convert("RGB")) for img in examples["image"]]
        return {"pixel_values": pixel_values, "lat": examples["latitude"], "lon": examples["longitude"]}

    ds.set_transform(preprocess)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_embeddings = []
    all_coords = []

    print("Generating reference embeddings...")
    with torch.no_grad():
        for batch in loader:
            imgs = batch["pixel_values"].to(DEVICE)
            lats = batch["lat"]
            lons = batch["lon"]

            emb = model(imgs)
            all_embeddings.append(emb.cpu().numpy())
            all_coords.append(np.stack([lats, lons], axis=1))

    np.save("ref_embeddings.npy", np.vstack(all_embeddings))
    np.save("ref_coords.npy", np.vstack(all_coords))
    print("Database saved: ref_embeddings.npy and ref_coords.npy")

if __name__ == "__main__":
    save_embeddings()
    ref_coords = np.load("ref_coords.npy")

    MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 39.950375, 39.95277777777778, -75.19258888888889, -75.17618611111112

    def get_label(lat, lon):
        row = int(((lat - MIN_LAT) / (MAX_LAT - MIN_LAT)) * BINS)
        col = int(((lon - MIN_LON) / (MAX_LON - MIN_LON)) * BINS)
        row = max(0, min(row, BINS - 1))
        col = max(0, min(col, BINS - 1))
        return row + (col * BINS)

    ref_bins = np.array([get_label(c[0], c[1]) for c in ref_coords])
    np.save("ref_bins.npy", ref_bins)
    print("Reference bins saved!")