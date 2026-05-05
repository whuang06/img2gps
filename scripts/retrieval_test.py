import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import math
import os
from datasets import load_dataset
from model import ResnetClassifier
from model import ViTClassifier

BINS = 3
TOP_K = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 39.950375, 39.95277777777778, -75.19258888888889, -75.17618611111112

DATASET_PATH = "Willh96/image2gps_compressed_dataset"
# REF_EMB_PATH = "/home/ec2-user/image2gps/scripts/ref_embeddings.npy"
# REF_CRD_PATH = "/home/ec2-user/image2gps/scripts/ref_coords.npy"
REF_EMB_PATH = "/home/ec2-user/image2gps/scripts/ref_embeddings_external.npy"
REF_CRD_PATH = "/home/ec2-user/image2gps/scripts/ref_coords_external.npy"
MODEL_PATH = "/home/ec2-user/image2gps/scripts/5_4_3bins_model_randomresize_ratio0.75_scale0.7_finetuned.pth"
MODEL_TYPE = "resnet"


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_bin_label(lat, lon):
    row = int(((lat - MIN_LAT) / (MAX_LAT - MIN_LAT)) * BINS)
    col = int(((lon - MIN_LON) / (MAX_LON - MIN_LON)) * BINS)
    row = max(0, min(row, BINS - 1))
    col = max(0, min(col, BINS - 1))
    return row + (col * BINS)

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        if MODEL_TYPE == "resnet":
            self.backbone = nn.Sequential(*list(original_model.resnet.children())[:-1])
            self._kind = "resnet"
        elif MODEL_TYPE == "vit":
            self.vit = original_model.vit
            self._kind = "vit"
        self.fc = original_model.resnet.fc if MODEL_TYPE == "resnet" else original_model.vit.heads.head

    def forward(self, x):
        if self._kind == "resnet":
            feat = torch.flatten(self.backbone(x), 1)
            logits = self.fc(feat)
            return feat, logits
        vit = self.vit
        x = vit._process_input(x)
        n = x.shape[0]
        cls = vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = vit.encoder(x)
        feat = x[:, 0]
        logits = self.fc(feat)
        return feat, logits


def evaluate():
    print("Loading reference database...")
    ref_embeddings = torch.from_numpy(np.load(REF_EMB_PATH)).to(DEVICE)
    ref_coords = np.load(REF_CRD_PATH)
    
    ref_bins = np.array([get_bin_label(c[0], c[1]) for c in ref_coords])

    print("Initializing Model...")
    if MODEL_TYPE == "resnet":
        base_model = ResnetClassifier(num_classes=BINS*BINS)
    elif MODEL_TYPE == "vit":
        base_model = ViTClassifier(num_classes=BINS*BINS)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = FeatureExtractor(base_model).to(DEVICE)
    model.eval()

    ds = load_dataset(DATASET_PATH)["test"]
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def preprocess(examples):
        pixel_values = [img_transforms(img.convert("RGB")) for img in examples["image"]]
        return {"pixel_values": pixel_values, "lat": examples["latitude"], "lon": examples["longitude"]}

    ds.set_transform(preprocess)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    global_errors = []
    local_errors = []

    print(f"Running comparison on {len(ds)} images...")
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch["pixel_values"].to(DEVICE)
            true_lats = batch["lat"]
            true_lons = batch["lon"]

            query_embs, logits = model(imgs)
            query_embs_norm = F.normalize(query_embs, p=2, dim=1)
            ref_embs_norm = F.normalize(ref_embeddings, p=2, dim=1)
            

            pred_bins = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(len(imgs)):
                t_lat, t_lon = true_lats[i], true_lons[i]
                q_feat = query_embs_norm[i].unsqueeze(0)

                # global retrieval
                global_sims = torch.mm(q_feat, ref_embs_norm.t())
                _, g_idx = torch.topk(global_sims, k=TOP_K)
                g_coords = ref_coords[g_idx[0].cpu().numpy()]
                g_pred_lat, g_pred_lon = np.median(g_coords[:, 0]), np.median(g_coords[:, 1])
                global_errors.append(haversine_distance(t_lat, t_lon, g_pred_lat, g_pred_lon))

                # local retrieval
                target_bin = pred_bins[i]
                mask = (ref_bins == target_bin)
                
                if not np.any(mask):
                    l_coords = g_coords 
                else:
                    l_embs_norm = ref_embs_norm[torch.from_numpy(mask).to(DEVICE)]
                    l_coords_raw = ref_coords[mask]
                    
                    local_sims = torch.mm(q_feat, l_embs_norm.t())
                    k_val = min(TOP_K, len(l_coords_raw))
                    _, l_idx = torch.topk(local_sims, k=k_val)
                    l_coords = l_coords_raw[l_idx[0].cpu().numpy()]
                
                l_pred_lat, l_pred_lon = np.median(l_coords[:, 0]), np.median(l_coords[:, 1])
                local_errors.append(haversine_distance(t_lat, t_lon, l_pred_lat, l_pred_lon))

    print("\n" + "="*50)
    print(f"{'METRIC':<20} | {'GLOBAL':<12} | {'LOCAL':<12}")
    print("-" * 50)
    print(f"{'Median Error (m)':<20} | {np.median(global_errors):<12.2f} | {np.median(local_errors):<12.2f}")
    print(f"{'Mean Error (m)':<20} | {np.mean(global_errors):<12.2f} | {np.mean(local_errors):<12.2f}")
    print(f"{'90th Percentile':<20} | {np.percentile(global_errors, 90):<12.2f} | {np.percentile(local_errors, 90):<12.2f}")
    print("="*50)

if __name__ == "__main__":
    evaluate()