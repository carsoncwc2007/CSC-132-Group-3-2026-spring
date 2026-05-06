# 
# Hand Gesture Control

# Control your mouse and volume using hand gestures via webcam.

# GESTURE MODE:
#   - ☝️  Index finger only  → Move mouse cursor
#   - ✌️  Index + Middle → Left click
#   - 🤏  Thumb + Index → Control volume (spread = louder)

# KEYBOARD SHORTCUTS:
#   - M   → Toggle between GESTURE and ASL mode
#   - ESC → Quit the program (key 27 )
# 

# Imports 
import cv2
import hand_detector2 as hdm       # Our custom hand detector class
import pyautogui                    # Mouse/cursor control
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ASL imports 
import pandas as pd
import time
from gtts import gTTS
import io
import pygame
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# Settings 
SMOOTHING = 5
CLICK_COOLDOWN = 20
VOLUME_MIN_DIST = 20
VOLUME_MAX_DIST = 200
CONFIDENCE_THRESHOLD = 0.7


# Screen Setup
screen_width, screen_height = pyautogui.size()
pyautogui.PAUSE = 0


# Hand Detector 
detector = hdm.handDetector(max_hands=1, detection_con=0.7, track_con=0.7)


# Volume Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]


data = pd.read_csv('hand_signals.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
X = data.drop('letter', axis=1)
y = data['letter']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
asl_model = LogisticRegression(max_iter=200)
asl_model.fit(X_train, y_train)



pygame.mixer.init()


# ── Webcam 
cap = cv2.VideoCapture(0)


# ── State Variables
mode = "GESTURE"
click_delay = 0
cursor_history_x = []
cursor_history_y = []
current_volume_pct = 50
fingers_display = [0, 0, 0, 0, 0]

# ASL state 
signal_data = {}
letters = [0]
word = ''
words = []
asl_start = time.time()
asl_end = time.time()



def speech(text):
    '''
    Converts a piece of text into speech using pyGame and gTTS libraries

    Parameters:
    text (string): A string representing the text to be converted into speech

    Returns: None
    '''
    myobj = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    myobj.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)



def draw_text_box(frame, text):
    """
    Draws the translated text on the screen.
    """
    height, width, _ = frame.shape
    cv2.rectangle(frame, (20, height - 90), (width - 20, height - 20), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Translation: {text}",
        (40, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# fingers up 
def fingers_up(lmlist):
    """
    Takes the landmark list from hand_detector2 (each item is [id, x, y])
    Returns [thumb, index, middle, ring, pinky] where 1=up, 0=down.
    """
    fingers = []
    # Thumb: check x position (left/right instead of up/down)
    fingers.append(1 if lmlist[4][1] < lmlist[3][1] else 0)
    # Other 4 fingers: check y position of tip vs two joints below
    for tip_id in [8, 12, 16, 20]:
        fingers.append(1 if lmlist[tip_id][2] < lmlist[tip_id - 2][2] else 0)
    return fingers


# smooth cursor 
def smooth_cursor(new_x, new_y):
    cursor_history_x.append(new_x)
    cursor_history_y.append(new_y)
    if len(cursor_history_x) > SMOOTHING:
        cursor_history_x.pop(0)
        cursor_history_y.pop(0)
    return int(np.mean(cursor_history_x)), int(np.mean(cursor_history_y))


# gesture mode label at top right
def draw_gesture_label(img):
    """Draws 'Mode: GESTURE' in the top right corner using a serif font."""
    h, w = img.shape[:2]
    text = "Mode: GESTURE"
    font = cv2.FONT_HERSHEY_COMPLEX   # closest to Times New Roman in OpenCV
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 150)             # green
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = w - text_size[0] - 15   # 15px padding from right edge
    text_y = 35                       # 35px from top
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)


#  ASL mode label at top right
def draw_asl_label(img):
    """Draws 'Mode: ASL' in the top right corner using the original font."""
    h, w = img.shape[:2]
    text = "Mode: ASL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 100, 255)             # blue
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = w - text_size[0] - 15
    text_y = 35
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)


# ── Main Loop 
print("Starting...Sujal's Gesture control Press M to toggle modes, ESC to quit.")

while True:
    success, img = cap.read()
    if not success:
        print("Warning: Could not read from webcam.")
        continue

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    key = cv2.waitKey(1) & 0xFF

    # Both modes use hand_detector2 for tracking
    img = detector.find_hands(img, draw=True)
    landmarks = detector.find_position(img, draw=False)

    
    # GESTURE MODE
    
    if mode == "GESTURE":

        if landmarks:
            label, lmlist = landmarks[0]

            index_tip_x = lmlist[8][1]
            index_tip_y = lmlist[8][2]
            thumb_tip_x = lmlist[4][1]
            thumb_tip_y = lmlist[4][2]

            fingers = fingers_up(lmlist)
            fingers_display = fingers

            #  CURSOR CONTROL — Index up, middle down
            if fingers[1] == 1 and fingers[2] == 0:
                raw_x = np.interp(index_tip_x, [0, w], [0, screen_width])
                raw_y = np.interp(index_tip_y, [0, h], [0, screen_height])
                smooth_x, smooth_y = smooth_cursor(raw_x, raw_y)
                pyautogui.moveTo(smooth_x, smooth_y)
                cv2.circle(img, (index_tip_x, index_tip_y), 12, (255, 100, 0), -1)
                cv2.putText(img, "MOVE", (index_tip_x + 15, index_tip_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            #  LEFT CLICK — Index + Middle up
            elif fingers[1] == 1 and fingers[2] == 1:
                if click_delay == 0:
                    pyautogui.click()
                    click_delay = CLICK_COOLDOWN
                cv2.circle(img, (index_tip_x, index_tip_y), 20, (0, 255, 255), 3)
                cv2.putText(img, "CLICK!", (index_tip_x + 15, index_tip_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 🔊VOLUME — Thumb and  Index finger distance
            distance = hypot(thumb_tip_x - index_tip_x, thumb_tip_y - index_tip_y)
            vol_level = np.interp(distance, [VOLUME_MIN_DIST, VOLUME_MAX_DIST], [vol_min, vol_max])
            volume.SetMasterVolumeLevel(vol_level, None)
            current_volume_pct = int(np.interp(distance, [VOLUME_MIN_DIST, VOLUME_MAX_DIST], [0, 100]))
            cv2.circle(img, (thumb_tip_x, thumb_tip_y), 10, (0, 200, 0), -1)
            cv2.circle(img, (index_tip_x, index_tip_y), 10, (0, 200, 0), -1)
            cv2.line(img, (thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y), (0, 200, 255), 2)

        else:
            cv2.putText(img, "Show your hand to the camera",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

        if click_delay > 0:
            click_delay -= 1

        draw_gesture_label(img)

     #ASL MODE:
     #Hold an ASL letter sign for 20 frames → adds letter to word
     #3 seconds of no hand → speaks the word aloud and resets
     #Press C → capture current landmark data to hand_signals.csv
    
    elif mode == "ASL":

        # No hand detected — run inactivity timer
        if not landmarks:
            asl_start = time.time()
            idle_timer = asl_start - asl_end

            # 3 seconds idle → speak the word
            if idle_timer >= 3 and word != '':
                if word[-1] != ' ':
                    speech(word)
                    words.append(word)
                    word = word + ' '

        # Hand detected
        if landmarks:
            label, lmlist = landmarks[0]
            fingers_display = fingers_up(lmlist)

            # Stop inactivity timer
            asl_end = time.time()

            # Draw bounding box around the hand
            p1 = (min(lmlist[x][1] for x in range(len(lmlist))) - 25,
                  min(lmlist[x][2] for x in range(len(lmlist))) - 25)
            p2 = (max(lmlist[x][1] for x in range(len(lmlist))) + 25,
                  max(lmlist[x][2] for x in range(len(lmlist))) + 25)
            cv2.rectangle(img, p1, p2, (255, 255, 255), 3)

            # Building the  location vector and predict letter
            location_vector = np.array([coord for lm in lmlist for coord in lm[1:3]]).reshape(1, -1)
            probabilities = asl_model.predict_proba(location_vector)
            max_prob = np.max(probabilities)

            if max_prob > CONFIDENCE_THRESHOLD:
                predicted_letter = asl_model.predict(location_vector)[0]
                if predicted_letter == letters[-1]:
                    letters.append(predicted_letter)
                else:
                    letters = [predicted_letter]
                cv2.putText(img, predicted_letter, (p1[0], p1[1] - 10),
                            cv2.QT_FONT_NORMAL, 3, (255, 255, 255), 3)

            # Same letter held for 20 frames → add to word
            if len(letters) == 20:
                word = word + letters[0]
                letters = [0]
                print(word)

            # Press C to capture new landmark data
            if key == ord('c'):
                for item in lmlist:
                    if f'{item[0]}x' in signal_data:
                        signal_data[f'{item[0]}x'].append(item[1])
                    else:
                        signal_data[f'{item[0]}x'] = [item[1]]
                    if f'{item[0]}y' in signal_data:
                        signal_data[f'{item[0]}y'].append(item[2])
                    else:
                        signal_data[f'{item[0]}y'] = [item[2]]

        draw_text_box(img, word.strip() if word.strip() else "Waiting...")
        draw_asl_label(img)

    # Show frame 
    cv2.imshow("Hand Gesture Control", img)

    # Toggle modes
    if key == ord('m'):
        mode = "ASL" if mode == "GESTURE" else "GESTURE"
        print(f"Switched to {mode} mode")
        # Reset ASL state when entering ASL mode
        if mode == "ASL":
            letters = [0]
            word = ''
            words = []
            asl_start = time.time()
            asl_end = time.time()

    elif key == ord('q') or key == 27:   # Q or ESC to quit
        print("Exiting...")
        break


# ── Save any captured landmark data before closing 
if signal_data:
    signal_data['letter'] = ['a'] * len(signal_data['0x'])
    new_signals = pd.DataFrame(signal_data)
    existing_signals = pd.read_csv('hand_signals.csv')
    updated_stats = pd.concat([existing_signals, new_signals], ignore_index=True)
    updated_stats.to_csv('hand_signals.csv', index=False)

cap.release()
cv2.destroyAllWindows()