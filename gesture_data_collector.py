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
    csv_filename= f'gesure_data/{letter_to_collect}.csv'
    
    #Initialize CSV file with headers
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as f:
            writer=csv.writer(f)
            headers=[f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label']
            # 63 normalized coordinates +label+confidence
            writer.writerow(headers)
    collected_count =0
    buffer_size=10
    frame_buffer=[]
    recording = False
    
    while collected_count < num_samples:
        ret,frame=cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame=cv2.flip(frame,1)
        rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=hands.process(rgb_frame)
        #display instructions
        status_text = f"letter:{letter_to_collect} | Collected: {collected_count}/{num_samples}"
        cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if recording:
            recording_text = f"Recording...({len(frame_buffer)}/{buffer_size})"
            cv2.putText(frame, recording_text, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv2.putText(frame, "Press SPACE to start recording", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Gesture Data Collector", frame)
        #handle key press
        key=cv2.waitKey(1) & 0xFF
        if key== ord('q'):
            break
        elif key == ord(' '):
            if results.multi_hand_landmarks and not recording:
                recording = True
                frame_buffer=[]
        #if recording, buffer frames
        if recording and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            confidence = get_detection_confidence(results)
            if hand_landmarks:
                feature_vector=normalize_landmarks(hand_landmarks)
                frame_buffer.append(feature_vector)
        #once buffer is full, process and save
        if recording and len(frame_buffer) == buffer_size:
            averaged_features=np.mean([fv[0]for fv in frame_buffer], axis=0)
            avg_confidence =np.mean([fv[1] for fv in frame_buffer])
            if is_valid_feature_vector(averaged_features):
                with open(csv_filename, mode='a', newline='') as f:
                    writer=csv.writer(f)
                    row=list(averaged_features) + [letter_to_collect] + [avg_confidence]
                    writer.writerow(row)
                print(f"saved sample {collected_count+1}/{num_samples} for letter '{letter_to_collect}")
                collected_count += 1
        recording = False
        frame_buffer = []
    cap.release()
    cv2.destroyAllWindows()
    print(f"finished collecting {collected_count} samples for letter '{letter_to_collect}'")
def collect_all_letters(num_samples=100):
    """collect data for all 26 letters"""
    letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for letter in letters:
        print(f"\n{'='*50}")
        print(f"Collecting data for letter '{letter}'")
        print(f"{'='*50}\n")
        collect_gesture_data(letter, num_samples)
if __name__ == "__main__":
    #collect data for a single letter (for testing)
    #collect_gesture_data('A', num_samples=10)
    # or collect data for all letters
    collect_all_letters(num_samples=100)
        