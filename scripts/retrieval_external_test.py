import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import math
import os
import pandas as pd
from PIL import Image
from model import ResnetClassifier
from model import ViTClassifier

BINS = 3
TOP_K = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 39.950375, 39.95277777777778, -75.19258888888889, -75.17618611111112

EXTERNAL_CSV = "/home/ec2-user/image2gps/playing_around_with_sample/sample_data/metadata.csv"
EXTERNAL_IMG_DIR = "/home/ec2-user/image2gps/playing_around_with_sample/sample_data"
# REF_EMB_PATH = "/home/ec2-user/image2gps/scripts/ref_embeddings.npy"
# REF_CRD_PATH = "/home/ec2-user/image2gps/scripts/ref_coords.npy"
REF_EMB_PATH = "/home/ec2-user/image2gps/scripts/ref_embeddings_external.npy"
REF_CRD_PATH = "/home/ec2-user/image2gps/scripts/ref_coords_external.npy"
MODEL_PATH = "/home/ec2-user/image2gps/scripts/5_4_3bins_model_randomresize_ratio0.75_scale0.7_finetuned.pth"
MODEL_TYPE = "resnet"


class ExternalTestDataset(Dataset):
    def __init__(self, csv_path, parent_dir, transform=None):
        self.metadata = pd.read_csv(csv_path)
        self.parent_dir = parent_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.parent_dir, row['image_file'])
        lat = row['Latitude']
        lon = row['Longitude']
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, lat, lon

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
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

def evaluate_external():
    print("Loading reference database...")
    ref_embeddings = torch.from_numpy(np.load(REF_EMB_PATH)).to(DEVICE)
    ref_coords = np.load(REF_CRD_PATH)
    ref_bins = np.array([get_bin_label(c[0], c[1]) for c in ref_coords])

    if MODEL_TYPE == "resnet":
        base_model = ResnetClassifier(num_classes=BINS*BINS)
    elif MODEL_TYPE == "vit":
        base_model = ViTClassifier(num_classes=BINS*BINS)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = FeatureExtractor(base_model).to(DEVICE)
    model.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_ds = ExternalTestDataset(EXTERNAL_CSV, EXTERNAL_IMG_DIR, transform=img_transforms)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    global_errs, local_errs = [], []

    print(f"Testing on {len(test_ds)} EXTERNAL samples...")
    with torch.no_grad():
        for imgs, t_lats, t_lons in loader:
            imgs = imgs.to(DEVICE)
            query_embs, logits = model(imgs)
            
            query_embs_norm = F.normalize(query_embs, p=2, dim=1)
            ref_embs_norm = F.normalize(ref_embeddings, p=2, dim=1)
            pred_bins = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(len(imgs)):
                q_feat = query_embs_norm[i].unsqueeze(0)
                true_lat, true_lon = t_lats[i].item(), t_lons[i].item()

                # global retrieval

                g_sims = torch.mm(q_feat, ref_embs_norm.t())
                _, g_idx = torch.topk(g_sims, k=TOP_K)
                g_pts = ref_coords[g_idx[0].cpu().numpy()]
                global_errs.append(haversine_distance(true_lat, true_lon, np.median(g_pts[:,0]), np.median(g_pts[:,1])))

                # local retrieval
                
                target_bin = pred_bins[i]
                mask = (ref_bins == target_bin)
                if not np.any(mask):
                    l_err = global_errs[-1]
                else:
                    l_embs_norm = ref_embs_norm[torch.from_numpy(mask).to(DEVICE)]
                    l_coords_raw = ref_coords[mask]
                    l_sims = torch.mm(q_feat, l_embs_norm.t())
                    k_val = min(TOP_K, len(l_coords_raw))
                    _, l_idx = torch.topk(l_sims, k=k_val)
                    l_pts = l_coords_raw[l_idx[0].cpu().numpy()]
                    l_err = haversine_distance(true_lat, true_lon, np.median(l_pts[:,0]), np.median(l_pts[:,1]))
                local_errs.append(l_err)

    print("\n" + "="*55)
    print(f"{'METRIC (EXTERNAL DATA)':<25} | {'GLOBAL':<12} | {'LOCAL':<12}")
    print("-" * 55)
    print(f"{'Median Error (m)':<25} | {np.median(global_errs):<12.2f} | {np.median(local_errs):<12.2f}")
    print(f"{'Mean Error (m)':<25} | {np.mean(global_errs):<12.2f} | {np.mean(local_errs):<12.2f}")
    print(f"{'Worst 10% (m)':<25} | {np.percentile(global_errs, 90):<12.2f} | {np.percentile(local_errs, 90):<12.2f}")
    print("="*55)

if __name__ == "__main__":
    evaluate_external()