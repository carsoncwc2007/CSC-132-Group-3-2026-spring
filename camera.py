import numpy as np
import cv2
import mediapipe as mp
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
print( "testing")
while True:
    ret, frame = cap.read()
    #applies the model
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=mp_hands.Hands().process(frame)
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
