import cv2
cap = cv2.VideoCapture("C:\\Users\\karan\\Desktop\\fakedetection\\data\\raw\\DFD_original_sequences\\01__exit_phone_room.mp4")

if not cap.isOpened():
    print("Failed to open video. Check FFmpeg or video format compatibility.")
else:
    print("Video opened successfully.")
