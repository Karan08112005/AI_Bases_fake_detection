from flask import Flask, render_template, request, jsonify
import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from training import create_model

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set device for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def extract_frames_from_video(video_path, frame_count=16):
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
    return frames

def predict(video_path, model):
    frames = extract_frames_from_video(video_path)
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

# Load model
model = load_model("model/model.pth")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload_and_predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded."})

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file."})

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    try:
        result = predict(video_path, model)
        os.remove(video_path)  # Clean up uploaded video
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
