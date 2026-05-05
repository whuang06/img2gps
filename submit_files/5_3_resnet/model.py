import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from typing import Any, Iterable, List

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.register_buffer("ref_embeddings", torch.zeros(6996, 512))
        self.register_buffer("ref_coords", torch.zeros(6996, 2))
        self.k = 10 

    def eval(self) -> None:
        self.backbone.eval()

    def predict(self, batch: Any) -> List[List[float]]:
        device = next(self.parameters()).device
        
        if not isinstance(batch, torch.Tensor):
            batch = torch.stack(list(batch))
        batch = batch.to(device)
        
        self.eval()
        with torch.no_grad():
            features = self.backbone(batch)
            query_embs = torch.flatten(features, 1)
            
            query_norm = F.normalize(query_embs, p=2, dim=1)
            ref_norm = F.normalize(self.ref_embeddings, p=2, dim=1)
            
            sim_matrix = torch.mm(query_norm, ref_norm.t())
            
            _, top_k_indices = torch.topk(sim_matrix, k=self.k, dim=1)
            
            predictions = []
            for i in range(query_embs.size(0)):
                neighbor_coords = self.ref_coords[top_k_indices[i]]
                median_lat_lon = torch.median(neighbor_coords, dim=0).values
                predictions.append(median_lat_lon.cpu().tolist())
                
        return predictions

def get_model() -> Model:
    return Model()