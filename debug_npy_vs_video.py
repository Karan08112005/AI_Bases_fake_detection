# debug_npy_vs_video.py
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Paths
npy_path = r"C:\Users\karan\Desktop\fakedetection\data\processed\tmp51ews_lu.npy"
video_path = r"C:\Users\karan\Desktop\fakedetection\data\raw\DFD_original_sequences\tmp51ews_lu.mp4"

# Load .npy file
frames_from_npy = np.load(npy_path)
print(f"Shape of frames from .npy: {frames_from_npy.shape}")

# Display a frame from .npy
plt.imshow(frames_from_npy[0])  # Display the first frame
plt.title("First Frame from .npy")
plt.show()

# Extract frames from .mp4 file
cap = cv2.VideoCapture(video_path)
frames_from_video = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to match preprocessing
    frames_from_video.append(frame_resized)

cap.release()
frames_from_video = np.array(frames_from_video)
print(f"Shape of frames from .mp4: {frames_from_video.shape}")

# Display a frame from the video
plt.imshow(cv2.cvtColor(frames_from_video[0], cv2.COLOR_BGR2RGB))  # Display first frame
plt.title("First Frame from .mp4")
plt.show()
