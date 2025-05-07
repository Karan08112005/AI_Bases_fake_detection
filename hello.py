import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

# DEVICE Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################
# Preprocessing.py
###############################################
def extract_frames(video_path, max_frames=30):
    """
    Extract frames from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No valid frames extracted from video: {video_path}")

    return np.array(frames)

def preprocess_dataset(data_dir, output_dir):
    """
    Preprocess a dataset of videos.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    categories = {
        "DFD_original_sequences": "REAL",
        "DFD_manipulated_sequences": "FAKE",
    }

    for category, label in categories.items():
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"Category path not found: {category_path}")
            continue

        for video_file in tqdm(os.listdir(category_path), desc=f"Processing {label} videos"):
            if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_path = os.path.join(category_path, video_file)
            if not os.path.isfile(video_path):
                continue

            try:
                frames = extract_frames(video_path)
                output_file = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.npy")
                np.save(output_file, frames)
                metadata.append({"file": output_file, "label": label})
            except ValueError as e:
                print(f"Skipping {video_file} due to error: {e}")

    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = os.path.join(output_dir, "processed_metadata.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"Metadata saved to {metadata_csv_path}")

###############################################
# Dataset and Model
###############################################
class DeepFakeDataset(Dataset):
    def __init__(self, metadata_csv, transform=None):
        self.data = pd.read_csv(metadata_csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frames = np.load(row['file'])
        label = 1 if row['label'] == 'FAKE' else 0
        aggregated_frame = np.mean(frames, axis=0).astype(np.uint8)

        if self.transform:
            aggregated_frame = self.transform(aggregated_frame)

        return aggregated_frame, label

def create_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

###############################################
# Training.py
###############################################
def train_model(data_dir, model_save_path, num_epochs=10, batch_size=32, learning_rate=1e-4):
    """
    Train the deepfake detection model.
    """
    metadata_csv = os.path.join(data_dir, "processed_metadata.csv")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DeepFakeDataset(metadata_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = create_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss / len(dataloader)}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

###############################################
# Deployment.py
###############################################
def load_model(model_path):
    """
    Load the trained model.
    """
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict(video_path, model):
    """
    Predict if a video is REAL or FAKE.
    """
    frames = extract_frames(video_path)
    aggregated_frame = np.mean(frames, axis=0).astype(np.uint8)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(aggregated_frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, prediction = torch.max(outputs, 1)

    return "FAKE" if prediction.item() == 1 else "REAL"

if __name__ == "__main__":
    # Example Usage
    data_dir = r"C:\Users\karan\Desktop\fakedetection\data\raw"
    output_dir = r"C:\Users\karan\Desktop\fakedetection\data\processed"
    model_save_path = r"C:\Users\karan\Desktop\fakedetection\models\best_model.pth"
    test_video_path = r"C:\Users\karan\Downloads\download.mp4"

    # Preprocess Dataset
    preprocess_dataset(data_dir, output_dir)

    # Train Model
    train_model(output_dir, model_save_path, num_epochs=5)

    # Load Model and Predict
    model = load_model(model_save_path)
    result = predict(test_video_path, model)
    print(f"Prediction for video '{test_video_path}': {result}")
