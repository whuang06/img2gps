import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from datasets import load_dataset
from datasets import load_from_disk
from model import ResnetClassifier

BINS = 3
EPOCHS = 15
BATCH_SIZE = 256
DROPOUT_P = 0.5
model_path = "5_4_3bins_model_colorjitter.pth"
# SEED = 42

def get_bounds(dataset_dict):
    lats = []
    lons = []
    for split in dataset_dict.keys():
        lats.extend(dataset_dict[split]["latitude"])
        lons.extend(dataset_dict[split]["longitude"])
    
    return min(lats), max(lats), min(lons), max(lons)

def get_label(lat, lon, min_lat, max_lat, min_lon, max_lon, bins):
    row = int(((lat - min_lat) / (max_lat - min_lat)) * bins)
    col = int(((lon - min_lon) / (max_lon - min_lon)) * bins)
    row = max(0, min(row, bins - 1))
    col = max(0, min(col, bins - 1))
    return row + (col * bins)

def train_model(model, epochs, train_loader, val_loader, optimizer, scheduler, criterion, device):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved at epoch {epoch+1}")
        val_acc = val_correct / val_total

        test_correct = 0
        test_total = 0
        running_test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (preds == labels).sum().item()
        avg_test_loss = running_test_loss / len(test_loader)
        test_acc = test_correct / test_total

        print(f"Epoch [{epoch+1}/{epochs}]")
        scheduler.step()
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}")
        print(f"  Test Loss:  {avg_test_loss:.4f} | Test Acc:   {test_acc:.2f}")
        print("-" * 30)

    return model

if __name__ == "__main__":
    ds = load_dataset("Willh96/image2gps_compressed_dataset")
    # ds = load_from_disk("/home/ec2-user/image2gps/preprocessed_augmented_dataset")
    min_lat, max_lat, min_lon, max_lon = get_bounds(ds)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 0.75)),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess_train(examples):
        pixel_values = [train_transforms(image.convert("RGB")) for image in examples["image"]]
        labels = [get_label(lat, lon, min_lat, max_lat, min_lon, max_lon, BINS) for lat, lon in zip(examples["latitude"], examples["longitude"])]
        return {"pixel_values": pixel_values, "label": labels}

    def preprocess_test(examples):
        pixel_values = [test_transforms(image.convert("RGB")) for image in examples["image"]]
        labels = [get_label(lat, lon, min_lat, max_lat, min_lon, max_lon, BINS) for lat, lon in zip(examples["latitude"], examples["longitude"])]
        return {"pixel_values": pixel_values, "label": labels}

    ds["train"].set_transform(preprocess_train)
    ds["validation"].set_transform(preprocess_test)
    ds["test"].set_transform(preprocess_test)

    def collate_fn(examples):
        images = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return images, labels

    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=14, pin_memory=True)
    val_loader = DataLoader(ds["validation"], batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=14, pin_memory=True)
    test_loader = DataLoader(ds["test"], batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=14, pin_memory=True)

    model = ResnetClassifier(num_classes=BINS * BINS, dropout_p=DROPOUT_P)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model = train_model(model, EPOCHS, train_loader, val_loader, optimizer, scheduler, criterion, device)

    print("Training Complete.")