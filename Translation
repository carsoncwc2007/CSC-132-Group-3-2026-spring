import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import time


class StablePrediction:
    """
    Keeps recent predictions and only confirms a sign when it appears consistently.
    This prevents the text from breaking/changing too fast after a few tests.
    """
    def __init__(self, max_history=12, min_confidence_count=7):
        self.history = deque(maxlen=max_history)
        self.min_confidence_count = min_confidence_count
        self.last_confirmed = "Waiting..."
        self.last_update_time = time.time()

    def update(self, prediction):
        if prediction is None:
            return self.last_confirmed

        self.history.append(prediction)

        counts = Counter(self.history)
        most_common_sign, count = counts.most_common(1)[0]

        if count >= self.min_confidence_count:
            self.last_confirmed = most_common_sign
            self.last_update_time = time.time()

        return self.last_confirmed


def get_finger_states(hand_landmarks, handedness_label):
    """
    Returns a list showing whether each finger is up.
    Order:
    [thumb, index, middle, ring, pinky]

    This is a simple rule-based system.
    """
    lm = hand_landmarks.landmark

    # Landmark tips and lower joints
    thumb_tip = lm[4]
    thumb_ip = lm[3]

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    fingers = []

    # Thumb logic depends on left or right hand
    if handedness_label == "Right":
        thumb_up = thumb_tip.x < thumb_ip.x
    else:
        thumb_up = thumb_tip.x > thumb_ip.x

    fingers.append(thumb_up)

    # Other fingers: tip higher than PIP joint means finger is up
    # In image coordinates, smaller y means higher
    for tip, pip in zip(finger_tips, finger_pips):
        fingers.append(lm[tip].y < lm[pip].y)

    return fingers


def classify_basic_sign(fingers):
    """
    Very basic demo signs.
    True means finger is up.
    False means finger is down.

    [thumb, index, middle, ring, pinky]
    """

    thumb, index, middle, ring, pinky = fingers

    if fingers == [False, False, False, False, False]:
        return "Fist"

    if fingers == [True, False, False, False, False]:
        return "Thumbs Up"

    if fingers == [False, True, False, False, False]:
        return "Pointing"

    if fingers == [False, True, True, False, False]:
        return "Peace"

    if fingers == [True, True, True, True, True]:
        return "Open Hand"

    if fingers == [False, True, True, True, True]:
        return "Four Fingers"

    if fingers == [False, False, False, False, True]:
        return "Pinky"

    return "Unknown Sign"


def draw_text_box(frame, text):
    """
    Draws the translated text on the screen.
    """
    height, width, _ = frame.shape

    # Background rectangle
    cv2.rectangle(frame, (20, height - 90), (width - 20, height - 20), (0, 0, 0), -1)

    # Main text
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


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    stable_prediction = StablePrediction(max_history=12, min_confidence_count=7)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Helps webcam stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Camera started.")
    print("Press Q to quit.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        ) as hands:

            while True:
                success, frame = cap.read()

                if not success or frame is None:
                    print("WARNING: Failed to read frame. Trying again...")
                    time.sleep(0.05)
                    continue

                # Flip camera so it acts like a mirror
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False

                results = hands.process(rgb_frame)

                rgb_frame.flags.writeable = True

                current_prediction = None

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):
                        handedness_label = handedness.classification[0].label

                        fingers = get_finger_states(hand_landmarks, handedness_label)
                        current_prediction = classify_basic_sign(fingers)

                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

                        # Show finger state for debugging
                        cv2.putText(
                            frame,
                            f"Fingers: {fingers}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                else:
                    current_prediction = "No Hand Detected"

                final_text = stable_prediction.update(current_prediction)

                draw_text_box(frame, final_text)

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Sign Language Translator", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")

    except Exception as error:
        print("ERROR: Something went wrong.")
        print(error)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed safely.")


if __name__ == "__main__":
    main()
