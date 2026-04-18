
import numpy as np
import cv2
import mediapipe as mp
#importing libraries 
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
#mediapipe setup
cap = cv2.VideoCapture(0)
#opening the webcam
hands=mp_hands.Hands(
static_image_mode=False,
max_num_hands=2,
min_detection_confidence=0.5,
min_tracking_confidence=0.5
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame=cv2.flip(frame,1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=hands.process(rgb_frame)
    #annotations the image
    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks,connections=mp_hands.HAND_CONNECTIONS)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
