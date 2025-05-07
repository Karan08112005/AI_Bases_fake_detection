import os
import cv2
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_frames(video_path, num_frames=10, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        logging.warning(f"Video {video_path} has no frames, skipping.")
        cap.release()
        return None

    frames = []
    interval = max(1, frame_count // num_frames)

    try:
        for i in range(min(num_frames, frame_count)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Unable to read frame {i} from {video_path}")
                break
            frames.append(cv2.resize(frame, frame_size))
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")
    finally:
        cap.release()

    if len(frames) > 0:
        return np.array(frames)
    else:
        logging.warning(f"No frames extracted from {video_path}")
        return None

def preprocess_dataset(raw_data_path, processed_data_path, num_frames=10, frame_size=(224, 224)):
    metadata = []
    os.makedirs(processed_data_path, exist_ok=True)

    logging.info(f"Processing directory: {raw_data_path}")
    
    for root, dirs, files in os.walk(raw_data_path):

        for file in files:
            file_path = os.path.join(root, file)

            if not file.lower().endswith(('.mp4', '.avi', '.mov')):
                logging.info(f"Skipping non-video file: {file}")
                continue

            logging.info(f"Processing video: {file_path}")
            frames = extract_frames(file_path, num_frames=num_frames, frame_size=frame_size)

            if frames is not None:
                npy_file_name = os.path.splitext(os.path.relpath(file_path, raw_data_path))[0].replace(os.sep, '_') + ".npy"
                npy_path = os.path.join(processed_data_path, npy_file_name)

                np.save(npy_path, frames)
                label = "FAKE" if "fake" in file.lower() else "REAL"
                metadata.append({"file": npy_path, "label": label})
            else:
                logging.info(f"Skipping {file}: No frames extracted.")

    if metadata:
        metadata_df = pd.DataFrame(metadata)
        metadata_csv_path = os.path.join(processed_data_path, "processed_metadata.csv")
        metadata_df.to_csv(metadata_csv_path, index=False)
        logging.info(f"Metadata saved to {metadata_csv_path}")
    else:
        logging.warning("No valid videos found. Metadata file will not be created.")

if __name__ == "__main__":
    raw_data_path = r"C:\Users\karan\Desktop\fakedetection\data\raw"
    processed_data_path = r"C:\Users\karan\Desktop\fakedetection\data\processed"

    logging.info("Starting preprocessing...")
    preprocess_dataset(raw_data_path, processed_data_path)
    logging.info("Preprocessing completed.")
