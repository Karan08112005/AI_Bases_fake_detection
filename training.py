import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class
class DeepfakeDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        frames = np.load(row['file'])
        label = 1 if row['label'] == 'FAKE' else 0

        # Aggregate frames by taking the mean
        aggregated_frame = np.mean(frames, axis=0).astype(np.uint8)

        # Convert to channel-first format
        aggregated_frame = np.transpose(aggregated_frame, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        # Convert to Tensor before applying transforms
        aggregated_frame = torch.tensor(aggregated_frame, dtype=torch.float32) / 255.0

        # Apply transformations if provided
        if self.transform:
            aggregated_frame = self.transform(aggregated_frame)

        return aggregated_frame, label

# Function to extract mean features from frames
def extract_mean_features(frames):
    return np.mean(frames, axis=(0, 1, 2))

# Function to balance the dataset using SMOTE
def balance_dataset(metadata):
    file_paths = metadata['file']
    labels = metadata['label'].apply(lambda x: 1 if x == 'FAKE' else 0)

    # Extract meaningful features
    feature_extracted_data = []
    for file_path in file_paths:
        frames = np.load(file_path)  # Load the frame data
        features = extract_mean_features(frames)  # Extract meaningful features
        feature_extracted_data.append(features)

    feature_extracted_data = np.array(feature_extracted_data)

    smote = SMOTE(random_state=42)
    resampled_features, resampled_labels = smote.fit_resample(feature_extracted_data, labels)

    resampled_metadata = pd.DataFrame({
        'file': [file_paths.iloc[i] for i in range(len(resampled_features))],
        'label': ['FAKE' if label == 1 else 'REAL' for label in resampled_labels]
    })

    return resampled_metadata

# Model Creation Function
def create_model(fine_tune=True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if fine_tune:
        for param in model.features.parameters():
            param.requires_grad = False  # Freeze early layers

    # Modify the final classifier layer for binary classification
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),  # Dropout with 30%
        nn.Linear(model.classifier[1].in_features, 2)
    )
    return model

# Training Function
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, save_dir="models"):
    model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience = 5
    patience_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects = 0

            all_labels = []
            all_preds = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val':
                scheduler.step(epoch_acc)
                print("Classification Report:")
                print(classification_report(all_labels, all_preds, labels=[0, 1], zero_division=0))
                print("Confusion Matrix:")
                print(confusion_matrix(all_labels, all_preds, labels=[0, 1]))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        model.load_state_dict(best_model_wts)
                        return model

    print("Training complete.")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# Define transforms for dataset
def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(224, padding=10),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# Main script
if __name__ == "__main__":
    # Load metadata
    metadata = pd.read_csv(r"C:\Users\karan\Desktop\fakedetection\data\processed\processed_metadata.csv")

    # Balance the dataset
    metadata = balance_dataset(metadata)

    # Split into train and validation datasets
    train_metadata, val_metadata = train_test_split(metadata, test_size=0.2, stratify=metadata['label'], random_state=42)

    # Create datasets and dataloaders
    train_dataset = DeepfakeDataset(train_metadata, transform=get_transforms(augment=True))
    val_dataset = DeepfakeDataset(val_metadata, transform=get_transforms(augment=False))

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4),
    }

    # Define model, criterion, and optimizer
    model = create_model(fine_tune=True)
    class_weights = torch.tensor([0.5, 0.5]).to(DEVICE)  # Class weights for binary class
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train the model
    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=20, device=DEVICE)

    # Save the best model
    os.makedirs(r"C:\Users\karan\Desktop\fakedetection\models", exist_ok=True)
    torch.save(trained_model.state_dict(), r"C:\Users\karan\Desktop\fakedetection\models\best_model.pth")
