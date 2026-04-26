import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Load HandLandmarker

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# Webcam

cap = cv2.VideoCapture(0)
timestamp = 0

label = input("Enter label (A/B/C/L/Y/HELLO/YES/NO): ")

with open("asl_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

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


        # Save landmarks

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            row = []
            for point in lm:
                row.extend([point.x, point.y, point.z])

            row.append(label)
            writer.writerow(row)

            # draw points
            for point in lm:
                x = int(point.x * w)
                y = int(point.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.putText(frame, f"Collecting: {label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()