from preprocessing import preprocess_dataset
from training import train_model, DeepfakeDataset, create_model
from deployment import load_model, predict
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from config import DEVICE
import cv2
import os

if __name__ == "__main__":
    # Preprocessing
    preprocess_dataset("data/raw", "data/processed")

    # Training
    metadata = pd.read_csv("data/processed/processed_metadata.csv")
    train_meta, val_meta = train_test_split(metadata, test_size=0.2, random_state=42)

    train_dataset = DeepfakeDataset(train_meta)
    val_dataset = DeepfakeDataset(val_meta)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False)
    }

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, dataloaders, criterion, optimizer, 10, DEVICE)

    # Deployment
    model = load_model("models/model_epoch_10.pth")
    print(predict(r"C:\Users\karan\Downloads\647aedfc.mp4", model))

    # Video Handling
    video_path = "data/raw/DFD_original_sequences/tmp51ews_lu.mp4"
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
        else: 
            ret, frame = cap.read()
            if not ret:
                print(f"No valid frames extracted from video: {video_path}")
            else:
                print("Frame extracted successfully.")
