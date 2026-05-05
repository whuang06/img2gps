import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from model import ResnetClassifier

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
BINS = 3
MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 39.950375, 39.95277777777778, -75.19258888888889, -75.17618611111112
model_path = "/home/ec2-user/image2gps/scripts/5_4_3bins_model_randomresize_ratio0.75_scale0.7.pth"
model_output_path = "/home/ec2-user/image2gps/scripts/5_4_3bins_model_randomresize_ratio0.75_scale0.7_finetuned.pth"

class PhillyGPSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def get_bin(self, lat, lon):
        lat_idx = int((lat - MIN_LAT) / (MAX_LAT - MIN_LAT) * BINS)
        lon_idx = int((lon - MIN_LON) / (MAX_LON - MIN_LON) * BINS)
        lat_idx = max(0, min(BINS - 1, lat_idx))
        lon_idx = max(0, min(BINS - 1, lon_idx))
        return lat_idx * BINS + lon_idx

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_file'])
        image = Image.open(img_path).convert("RGB")
        
        label = self.get_bin(row['Latitude'], row['Longitude'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model(num_classes):
    model = ResnetClassifier(num_classes=num_classes)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(BINS * BINS).to(device)

full_dataset = PhillyGPSDataset(
    "/home/ec2-user/image2gps/playing_around_with_sample/sample_data/metadata.csv",
    "/home/ec2-user/image2gps/playing_around_with_sample/sample_data",
    transform=train_transforms,
)

train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=14, pin_memory=True)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    best_loss = float('inf')
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    avg_loss = loss.item()/len(train_loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_output_path)
        print(f"Model saved to {model_output_path}")
    print(f"Epoch {epoch+1} complete. Loss: {avg_loss:.4f}")