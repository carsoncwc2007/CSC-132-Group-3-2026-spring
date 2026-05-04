import cv2
import mediapipe as mp
from collections import deque, Counter
import math
import time


class StablePrediction:
    """
    Keeps recent predictions and only confirms a letter when it appears consistently.
    This helps stop the translation from flickering too much.
    """

    def __init__(self, max_history=15, min_confidence_count=8):
        self.history = deque(maxlen=max_history)
        self.last_confirmed = "Waiting..."
        self.min_confidence_count = min_confidence_count

    def update(self, prediction):
        if prediction is None:
            return self.last_confirmed

        self.history.append(prediction)

        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]

        if count >= self.min_confidence_count:
            self.last_confirmed = most_common

        return self.last_confirmed


class MotionTracker:
    """
    Tracks fingertip movement for motion letters.
    J uses the pinky fingertip.
    Z uses the index fingertip.
    """

    def __init__(self, max_points=25):
        self.index_path = deque(maxlen=max_points)
        self.pinky_path = deque(maxlen=max_points)

    def update(self, hand_landmarks):
        lm = hand_landmarks.landmark

        index_tip = lm[8]
        pinky_tip = lm[20]

        self.index_path.append((index_tip.x, index_tip.y))
        self.pinky_path.append((pinky_tip.x, pinky_tip.y))

    def clear(self):
        self.index_path.clear()
        self.pinky_path.clear()

    def detect_j(self):
        """
        Rough J motion:
        pinky moves downward, then hooks sideways/upward.
        """
        if len(self.pinky_path) < 12:
            return False

        points = list(self.pinky_path)

        start_x, start_y = points[0]
        mid_x, mid_y = points[len(points) // 2]
        end_x, end_y = points[-1]

        moved_down = mid_y > start_y + 0.08
        hooked_sideways = abs(end_x - mid_x) > 0.06
        hooked_up = end_y < mid_y - 0.03

        return moved_down and hooked_sideways and hooked_up

    def detect_z(self):
        """
        Rough Z motion:
        index finger moves side, diagonal down, side.
        """
        if len(self.index_path) < 15:
            return False

        points = list(self.index_path)

        start_x, start_y = points[0]
        one_third_x, one_third_y = points[len(points) // 3]
        two_third_x, two_third_y = points[(2 * len(points)) // 3]
        end_x, end_y = points[-1]

        first_side = (
            abs(one_third_x - start_x) > 0.06
            and abs(one_third_y - start_y) < 0.08
        )

        diagonal_down = (
            abs(two_third_x - one_third_x) > 0.05
            and two_third_y > one_third_y + 0.05
        )

        second_side = (
            abs(end_x - two_third_x) > 0.06
            and abs(end_y - two_third_y) < 0.08
        )

        return first_side and diagonal_down and second_side


def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


def get_finger_states(hand_landmarks, handedness_label):
    """
    Returns which fingers are up.

    Order:
    [thumb, index, middle, ring, pinky]
    """
    lm = hand_landmarks.landmark

    fingers = []

    # Thumb detection
    if handedness_label == "Right":
        thumb_up = lm[4].x < lm[3].x
    else:
        thumb_up = lm[4].x > lm[3].x

    fingers.append(thumb_up)

    # Other fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        fingers.append(lm[tip].y < lm[pip].y)

    return fingers


def classify_asl_letter(hand_landmarks, handedness_label):
    """
    Basic rule-based ASL alphabet classifier.

    This is a starter version, not a perfect full ASL model.
    """
    lm = hand_landmarks.landmark
    fingers = get_finger_states(hand_landmarks, handedness_label)

    thumb, index, middle, ring, pinky = fingers

    thumb_tip = lm[4]
    index_tip = lm[8]
    middle_tip = lm[12]
    ring_tip = lm[16]
    pinky_tip = lm[20]

    index_mcp = lm[5]
    middle_mcp = lm[9]
    ring_mcp = lm[13]
    pinky_mcp = lm[17]

    wrist = lm[0]

    thumb_index_dist = distance(thumb_tip, index_tip)
    thumb_middle_dist = distance(thumb_tip, middle_tip)
    thumb_ring_dist = distance(thumb_tip, ring_tip)
    thumb_pinky_dist = distance(thumb_tip, pinky_tip)

    index_middle_dist = distance(index_tip, middle_tip)

    # A: fist with thumb on side
    if fingers == [True, False, False, False, False]:
        return "A"

    # B: four fingers up, thumb tucked
    if fingers == [False, True, True, True, True]:
        return "B"

    # C: curved hand shape
    if (
        not index and not middle and not ring and not pinky
        and 0.08 < thumb_index_dist < 0.25
    ):
        return "C"

    # D: index up
    if index and not middle and not ring and not pinky:
        return "D"

    # E: fingers bent down, thumb close
    if fingers == [False, False, False, False, False] and thumb_index_dist < 0.12:
        return "E"

    # F: index and thumb touch, other fingers up
    if thumb_index_dist < 0.06 and middle and ring and pinky:
        return "F"

    # G: index points sideways, thumb out
    if index and not middle and not ring and not pinky and thumb:
        if abs(index_tip.y - index_mcp.y) < 0.12:
            return "G"

    # H: index and middle sideways together
    if index and middle and not ring and not pinky:
        if index_middle_dist < 0.08 and abs(index_tip.y - index_mcp.y) < 0.15:
            return "H"

    # I: pinky up
    if not index and not middle and not ring and pinky:
        return "I"

    # K: index and middle up, thumb near middle
    if index and middle and not ring and not pinky and thumb:
        if thumb_middle_dist < 0.12:
            return "K"

    # L: thumb and index out
    if thumb and index and not middle and not ring and not pinky:
        return "L"

    # M: closed fist-like shape
    if fingers == [False, False, False, False, False]:
        if thumb_tip.x > index_mcp.x and thumb_tip.x < pinky_mcp.x:
            return "M"

    # N: closed fist-like shape
    if fingers == [False, False, False, False, False]:
        if thumb_tip.x > index_mcp.x and thumb_tip.x < ring_mcp.x:
            return "N"

    # O: thumb and fingertips close in circle
    if (
        thumb_index_dist < 0.08
        and thumb_middle_dist < 0.12
        and thumb_ring_dist < 0.14
        and thumb_pinky_dist < 0.16
    ):
        return "O"

    # P: like K, but points downward
    if index and middle and not ring and not pinky and thumb:
        if index_tip.y > wrist.y:
            return "P"

    # Q: like G, but points downward
    if index and not middle and not ring and not pinky and thumb:
        if index_tip.y > wrist.y:
            return "Q"

    # R: index and middle close/crossed
    if index and middle and not ring and not pinky:
        if index_middle_dist < 0.04:
            return "R"

    # S: fist
    if fingers == [False, False, False, False, False]:
        return "S"

    # T: fist with thumb between index and middle
    if fingers == [False, False, False, False, False]:
        if distance(thumb_tip, index_mcp) < 0.10:
            return "T"

    # U: index and middle together
    if index and middle and not ring and not pinky:
        if index_middle_dist < 0.08:
            return "U"

    # V: index and middle spread apart
    if index and middle and not ring and not pinky:
        if index_middle_dist >= 0.08:
            return "V"

    # W: index, middle, ring up
    if not thumb and index and middle and ring and not pinky:
        return "W"

    # X: bent index hook
    if not thumb and not middle and not ring and not pinky:
        if lm[8].y > lm[6].y and lm[6].y < lm[5].y:
            return "X"

    # Y: thumb and pinky out
    if thumb and not index and not middle and not ring and pinky:
        return "Y"

    return "Unknown"


def draw_text_box(frame, text):
    """
    Keeps the original black text box style.
    Shows: Translation: LETTER
    """
    height, width, _ = frame.shape

    cv2.rectangle(
        frame,
        (20, height - 90),
        (width - 20, height - 20),
        (0, 0, 0),
        -1
    )

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

    stable_prediction = StablePrediction(max_history=15, min_confidence_count=8)
    motion_tracker = MotionTracker(max_points=25)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("ASL alphabet translator started.")
    print("Press Q to quit.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands:

            while True:
                success, frame = cap.read()

                if not success or frame is None:
                    print("WARNING: Could not read camera frame.")
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False

                results = hands.process(rgb_frame)

                rgb_frame.flags.writeable = True

                current_prediction = "No Hand Detected"

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):
                        handedness_label = handedness.classification[0].label

                        motion_tracker.update(hand_landmarks)

                        if motion_tracker.detect_j():
                            current_prediction = "J"
                            motion_tracker.clear()

                        elif motion_tracker.detect_z():
                            current_prediction = "Z"
                            motion_tracker.clear()

                        else:
                            current_prediction = classify_asl_letter(
                                hand_landmarks,
                                handedness_label
                            )

                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

                else:
                    current_prediction = "No Hand Detected"
                    motion_tracker.clear()

                final_translation = stable_prediction.update(current_prediction)

                draw_text_box(frame, final_translation)

                cv2.putText(
                    frame,
                    "ASL Alphabet Mode | Press Q to quit",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("ASL Alphabet Translator", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Program stopped by user.")

    except Exception as error:
        print("ERROR: Something went wrong.")
        print(error)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed safely.")


if __name__ == "__main__":
    main()
