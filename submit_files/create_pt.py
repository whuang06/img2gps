import torch
import numpy as np

trained_weights = torch.load("/home/ec2-user/image2gps/scripts/5_4_3bins_model_randomresize_ratio0.75_scale0.7_finetuned_hf.pth", map_location="cpu")
ref_embs = np.load("/home/ec2-user/image2gps/scripts/ref_embeddings.npy")
ref_crds = np.load("/home/ec2-user/image2gps/scripts/ref_coords.npy")

MODEL_TYPE = "resnet"

new_state_dict = {}

for key, value in trained_weights.items():
    if key.startswith(MODEL_TYPE):
        new_key = key.replace(MODEL_TYPE + ".", "backbone.")
        if "fc" not in new_key:
            new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

new_state_dict["ref_embeddings"] = torch.from_numpy(ref_embs).float()
new_state_dict["ref_coords"] = torch.from_numpy(ref_crds).float()

torch.save(new_state_dict, "model.pt")

print(f"Successfully bundled {len(ref_embs)} reference images into model.pt!")