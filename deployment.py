import torch
from torchvision import transforms
import numpy as np
import cv2
from preprocessing import extract_frames
from training import create_model

# Set device for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """
    Load the model from the given path.

    Args:
        model_path (str): Path to the saved model weights.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = create_model()

    # Ensure the model weights are loaded safely
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    except Exception as e:
        raise ValueError(f"Error loading model weights: {e}")

    model.to(DEVICE)
    model.eval()
    return model

def extract_frames_from_video(video_path, frame_count=16):
    """
    Extract frames evenly from the video.

    Args:
        video_path (str): Path to the video file.
        frame_count (int): Number of frames to extract.

    Returns:
        list: List of extracted frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(frame_count):
        frame_index = int(i * frame_total / frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be extracted from the video: {video_path}")

    return frames

def predict(video_path, model):
    """
    Predict if the video is REAL or FAKE using the trained model.

    Args:
        video_path (str): Path to the video file.
        model (torch.nn.Module): Trained model for inference.

    Returns:
        str: Prediction result ('REAL' or 'FAKE').
    """
    # Extract frames from the video
    try:
        frames = extract_frames_from_video(video_path)
    except Exception as e:
        raise ValueError(f"Error extracting frames: {e}")

    # Aggregate frames into a single representative image
    aggregated_frame = np.mean(frames, axis=0).astype(np.uint8)

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        # Transform the aggregated frame and prepare input tensor
        input_tensor = transform(aggregated_frame).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise ValueError(f"Error transforming frame: {e}")

    # Perform inference
    with torch.no_grad():
        try:
            outputs = model(input_tensor)
            _, prediction = torch.max(outputs, 1)
        except Exception as e:
            raise ValueError(f"Error during model inference: {e}")

    return "FAKE" if prediction.item() == 1 else "REAL"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video Real/Fake Detection")
    parser.add_argument("--model", required=True, help="Path to the trained model file.")
    parser.add_argument("--video", required=True, help="Path to the video file for prediction.")

    args = parser.parse_args()

    model_path = args.model
    video_path = args.video

    try:
        # Load the model
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully.")

        # Make a prediction
        print(f"Predicting for video: {video_path}")
        result = predict(video_path, model)
        print(f"Prediction for video '{video_path}': {result}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
