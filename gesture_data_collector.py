import cv2
import mediapipe as mp
import numpy as np 
import csv 
import os 
import time
from hand_utils import is_valid_feature_vector, normalize_landmarks, extract_single_hand, get_detection_confidence

os.makedirs('gesture_data', exist_ok=True)

def collect_gesture_data(letter_to_collect,num_samples=100):
    """
    ALGORITHM:
    1. Start webcam
    2. Show which letter user should demonstrate
    3. When user presses the letter key, start recording
    4. Capture N frames (e.g., 10 frames at 30 FPS = ~300ms of hand pose)
    5. Average the N frames to get single pose (reduces noise)
    6. Append to CSV file
    7. Repeat until num_samples collected
    
    WHY AVERAGE FRAMES:
    - Single frame can have noise/jitter from MediaPipe
    - Averaging 10 frames stabilizes the landmark position
    - User holds hand still for ~300ms to make single letter = 1 data point
    """
    mp_hands=mp.solutions.hands
    mp_drawing=mp.solutions.drawing_utils
    hands=mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap=cv2.VideoCapture(0)
    csv_filename
    