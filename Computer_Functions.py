import cv2
import mediapipe as mp
import pyautogui
import math
import time
import keyboard
import ctypes

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ===== AUDIO SETUP =====
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ===== HELPERS =====
def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def fingers_up(lm):
    fingers = []
    fingers.append(lm[4].x < lm[3].x)  # thumb
    fingers.append(lm[8].y < lm[6].y)  # index
    fingers.append(lm[12].y < lm[10].y) # middle
    fingers.append(lm[16].y < lm[14].y) # ring
    fingers.append(lm[20].y < lm[18].y) # pinky
    return fingers

last_action_time = 0
cooldown = 1.0
paused = False

prev_x = None

# ===== LOOP =====
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark

            finger_state = fingers_up(lm)

            thumb = lm[4]
            index = lm[8]
            middle = lm[12]

            thumb_index_dist = distance(thumb, index)
            index_middle_dist = distance(index, middle)

            # ===== PAUSE SYSTEM =====
            if finger_state == [True, True, True, True, True]:
                paused = True
                cv2.putText(img, "PAUSED", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                continue
            else:
                paused = False

            if paused:
                continue

            # ===== MOUSE MOVE =====
            if finger_state == [False, True, False, False, False]:
                screen_w, screen_h = pyautogui.size()
                x = int(index.x * screen_w)
                y = int(index.y * screen_h)
                pyautogui.moveTo(x, y)

            # ===== CLICK =====
            if index_middle_dist < 0.05 and current_time - last_action_time > cooldown:
                pyautogui.click()
                last_action_time = current_time

            # ===== VOLUME CONTROL =====
            vol = min(max(thumb_index_dist * 200 - 50, 0), 100)
            volume.SetMasterVolumeLevelScalar(vol / 100, None)

            # ===== THUMBS UP → PLAY/PAUSE =====
            if finger_state == [True, False, False, False, False] and current_time - last_action_time > cooldown:
                keyboard.press_and_release("play/pause media")
                last_action_time = current_time

            # ===== FIST → LOCK SCREEN =====
            if finger_state == [False, False, False, False, False] and current_time - last_action_time > cooldown:
                ctypes.windll.user32.LockWorkStation()
                last_action_time = current_time

            # ===== SWIPE GESTURES =====


            prev_x = index.x

    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
