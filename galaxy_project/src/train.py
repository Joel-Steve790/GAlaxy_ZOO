import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score

from model import GalaxyCNN


# Configuration
BATCH_SIZE = 16
EPOCHS = 20
IMAGE_SIZE = 128
LEARNING_RATE = 0.001

CLASS_NAMES = [
    "Smooth",
    "Spiral",
    "Disk_without_spiral"
]


# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# Device selection
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# Image transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


# Dataset loader
class GalaxyDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# Load datasets
train_dataset = GalaxyDataset(
    "prepared/train.csv",
    train_transform
)

val_dataset = GalaxyDataset(
    "prepared/val.csv",
    val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# Initialize model
model = GalaxyCNN(num_classes=3).to(device)


# Calculate class weights
counts = np.bincount(
    train_dataset.data["label"],
    minlength=3
)

weights = counts.sum() / (3 * counts)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# Training and validation
def run_epoch(loader, training):

    model.train(training)

    total_loss = 0
    predictions = []
    actual = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:

        for images, labels in tqdm(loader):

            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            predictions.extend(
                predicted.cpu().tolist()
            )

            actual.extend(
                labels.cpu().tolist()
            )

    accuracy = accuracy_score(actual, predictions)

    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    return total_loss / len(loader), accuracy, macro_f1


# Main training loop
best_f1 = 0

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

history = []

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    train_loss, train_acc, train_f1 = run_epoch(
        train_loader, True
    )

    val_loss, val_acc, val_f1 = run_epoch(
        val_loader, False
    )

    scheduler.step(val_f1)

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1
    })

    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE
            },
            "models/best_galaxy_model.pth"
        )

        print("Best model saved!")


pd.DataFrame(history).to_csv(
    "results/training_history.csv",
    index=False
)

print("\nTraining completed!")
print("Best validation F1:", best_f1)
