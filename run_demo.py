import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# LOAD MODEL

model = joblib.load("asl_model.pkl")


# LOAD MEDIAPIPE TASKS MODEL

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# MOTION + STABILITY BUFFERS

motion_buffer = deque(maxlen=10)
stable_buffer = deque(maxlen=5)

final_label = "Unknown"


# MOTION DETECTION

def detect_motion(buffer):
    if len(buffer) < 10:
        return None

    data = np.array(buffer)

    x_vals = data[:, 0]
    y_vals = data[:, 1]

    x_change = np.max(x_vals) - np.min(x_vals)
    y_change = np.max(y_vals) - np.min(y_vals)

    # HELLO = wave (left-right)
    if x_change > 0.15 and y_change < 0.1:
        return "HELLO (Wave)"

    # YES = nod (up-down)
    if y_change > 0.15 and x_change < 0.1:
        return "YES (Nod)"

    # NO = shake (strong movement both)
    if x_change > 0.2 and y_change > 0.2:
        return "NO (Shake)"

    return None


# WEBCAM

cap = cv2.VideoCapture(0)
timestamp = 0


# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 1
    result = detector.detect_for_video(mp_image, timestamp)

    label = "No Hand"


    # HAND PROCESSING
    
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        # feature vector for ML
        frame_features = []
        for point in lm:
            frame_features.extend([point.x, point.y, point.z])

        # motion tracking (wrist x,y)
        motion_buffer.append([lm[0].x, lm[0].y])


        # MOTION PRIORITY CHECK
        
        motion_label = detect_motion(motion_buffer)

        if motion_label:
            stable_buffer.append(motion_label)
        else:
            
            # STATIC ML MODEL
            
            prediction = model.predict([frame_features])[0]

            probs = model.predict_proba([frame_features])[0]
            confidence = probs.max()

            if confidence >= 0.60:
                stable_buffer.append(prediction)
            else:
                stable_buffer.append("Unknown")


        # STABILITY CHECK (5 FRAME MAJORITY VOTE)
        
        if len(stable_buffer) == stable_buffer.maxlen:
            most_common = max(set(stable_buffer), key=stable_buffer.count)

            if stable_buffer.count(most_common) >= 4:
                final_label = most_common
            else:
                final_label = "Unknown"


        # DRAW LANDMARKS
        
        for point in lm:
            x = int(point.x * w)
            y = int(point.y * h)
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            
    # UI
    
    cv2.putText(frame, "ASL Motion + ML Demo", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Output: {final_label}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.putText(frame, "Press Q to quit", (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()