import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from model import ResnetClassifier
from model import ViTClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BINS = 3
DATA_DIR = "/home/ec2-user/image2gps/playing_around_with_sample/sample_data"
CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
MODEL_PATH = "/home/ec2-user/image2gps/scripts/5_4_3bins_model_randomresize_ratio0.75_scale0.7_finetuned.pth"
MODEL_TYPE = "resnet"

OUT_EMB_PATH = "ref_embeddings_external.npy"
OUT_CRD_PATH = "ref_coords_external.npy"
OUT_BIN_PATH = "ref_bins_external.npy"


class ExternalImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_file"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        return image, lat, lon


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
        base_model = ResnetClassifier(num_classes=BINS * BINS)
    elif MODEL_TYPE == "vit":
        base_model = ViTClassifier(num_classes=BINS * BINS)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = FeatureExtractor(base_model).to(DEVICE)
    model.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = ExternalImageDataset(CSV_PATH, DATA_DIR, transform=img_transforms)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    all_embeddings = []
    all_coords = []

    print(f"Generating reference embeddings from {len(ds)} images in {DATA_DIR}...")
    with torch.no_grad():
        for imgs, lats, lons in loader:
            imgs = imgs.to(DEVICE)
            emb = model(imgs)
            all_embeddings.append(emb.cpu().numpy())
            coords = np.stack([lats.numpy(), lons.numpy()], axis=1)
            all_coords.append(coords)

    np.save(OUT_EMB_PATH, np.vstack(all_embeddings))
    np.save(OUT_CRD_PATH, np.vstack(all_coords))
    print(f"Database saved: {OUT_EMB_PATH} and {OUT_CRD_PATH}")


if __name__ == "__main__":
    save_embeddings()
    ref_coords = np.load(OUT_CRD_PATH)

    MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 39.950375, 39.95277777777778, -75.19258888888889, -75.17618611111112

    def get_label(lat, lon):
        row = int(((lat - MIN_LAT) / (MAX_LAT - MIN_LAT)) * BINS)
        col = int(((lon - MIN_LON) / (MAX_LON - MIN_LON)) * BINS)
        row = max(0, min(row, BINS - 1))
        col = max(0, min(col, BINS - 1))
        return row + (col * BINS)

    ref_bins = np.array([get_label(c[0], c[1]) for c in ref_coords])
    np.save(OUT_BIN_PATH, ref_bins)
    print(f"Reference bins saved to {OUT_BIN_PATH}!")
