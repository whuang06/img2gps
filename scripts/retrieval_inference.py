import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model import ResnetClassifier
import torch.nn.functional as F

ref_embeddings = torch.from_numpy(np.load("ref_embeddings.npy"))
ref_coords = np.load("ref_coords.npy")

def get_median_coords(query_img_path, k=10, bins=4):
    base_model = ResnetClassifier(num_classes=bins * bins)
    base_model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    backbone = nn.Sequential(*list(base_model.resnet.children())[:-1])
    backbone.eval()

    img = Image.open(query_img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    query_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        query_emb = torch.flatten(backbone(query_tensor), 1)

    similarities = F.cosine_similarity(query_emb, ref_embeddings)
    
    top_k_indices = torch.topk(similarities, k=k).indices.numpy()
    
    matched_coords = ref_coords[top_k_indices]
    median_lat = np.median(matched_coords[:, 0])
    median_lon = np.median(matched_coords[:, 1])

    return median_lat, median_lon, matched_coords

if __name__ == "__main__":
    test_img = "test_image.jpg"
    lat, lon, matches = get_median_coords(test_img, k=7)
    print(f"Predicted Median Coordinates: {lat}, {lon}")
    print(f"Top match distance spread: {np.ptp(matches, axis=0)}")