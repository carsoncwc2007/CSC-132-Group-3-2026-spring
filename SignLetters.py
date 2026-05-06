import cv2
import mediapipe as mp
from collections import deque, Counter
import math
import time


class StablePrediction:
    """
    Keeps recent predictions stable so the translation does not flicker too much.
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

    def clear(self):
        self.history.clear()
        self.last_confirmed = "Waiting..."


class MotionTracker:
    """
    Tracks fingertip movement for J and Z.

    J = pinky draws a J motion
    Z = index finger draws a Z motion
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


class SentenceBuilder:
    """
    Builds a sentence by locking in letters or commands.
    """

    def __init__(self, hold_time=1.0, cooldown=1.2):
        self.text = ""
        self.current_candidate = None
        self.candidate_start_time = None
        self.last_added_time = 0
        self.hold_time = hold_time
        self.cooldown = cooldown
        self.status = "Hold a letter steady to lock it in"

    def update(self, translation):
        now = time.time()

        ignored = ["Waiting...", "No Hand Detected", "Unknown"]

        if translation in ignored:
            self.current_candidate = None
            self.candidate_start_time = None
            self.status = "Show a letter or command"
            return self.text

        if translation != self.current_candidate:
            self.current_candidate = translation
            self.candidate_start_time = now
            self.status = f"Holding {translation}..."
            return self.text

        held_long_enough = now - self.candidate_start_time >= self.hold_time
        cooldown_finished = now - self.last_added_time >= self.cooldown

        if held_long_enough and cooldown_finished:
            self.add_translation(translation)
            self.last_added_time = now
            self.candidate_start_time = now
            self.status = f"Locked in: {translation}"

        return self.text

    def add_translation(self, translation):
        if translation == "SPACE":
            if len(self.text) > 0 and not self.text.endswith(" "):
                self.text += " "
            return

        if translation == "CLEAR":
            self.text = ""
            return

        if len(translation) == 1 and translation.isalpha():
            self.text += translation

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""


def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


def get_finger_states(hand_landmarks, handedness_label):
    """
    Returns finger states in this order:

    [thumb, index, middle, ring, pinky]
    """
    lm = hand_landmarks.landmark

    fingers = []

    if handedness_label == "Right":
        thumb_up = lm[4].x < lm[3].x
    else:
        thumb_up = lm[4].x > lm[3].x

    fingers.append(thumb_up)

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        fingers.append(lm[tip].y < lm[pip].y)

    return fingers


def is_open_hand(hand_landmarks, handedness_label):
    fingers = get_finger_states(hand_landmarks, handedness_label)
    return fingers == [True, True, True, True, True] or fingers == [False, True, True, True, True]


def is_fist(hand_landmarks, handedness_label):
    fingers = get_finger_states(hand_landmarks, handedness_label)
    return fingers == [False, False, False, False, False]


def detect_two_hand_command(hand_data):
    """
    Detects commands using two hands.

    Two open hands = SPACE
    Two fists = CLEAR
    """
    if len(hand_data) < 2:
        return None

    first_landmarks, first_label = hand_data[0]
    second_landmarks, second_label = hand_data[1]

    first_open = is_open_hand(first_landmarks, first_label)
    second_open = is_open_hand(second_landmarks, second_label)

    first_fist = is_fist(first_landmarks, first_label)
    second_fist = is_fist(second_landmarks, second_label)

    if first_open and second_open:
        return "SPACE"

    if first_fist and second_fist:
        return "CLEAR"

    return None


def classify_asl_letter(hand_landmarks, handedness_label):
    """
    Basic rule-based ASL alphabet classifier.

    A-Z are included.
    J and Z are handled separately with MotionTracker because they use movement.
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
    ring_mcp = lm[13]
    pinky_mcp = lm[17]

    wrist = lm[0]

    thumb_index_dist = distance(thumb_tip, index_tip)
    thumb_middle_dist = distance(thumb_tip, middle_tip)
    thumb_ring_dist = distance(thumb_tip, ring_tip)
    thumb_pinky_dist = distance(thumb_tip, pinky_tip)

    index_middle_dist = distance(index_tip, middle_tip)

    # A
    if fingers == [True, False, False, False, False]:
        return "A"

    # B
    if fingers == [False, True, True, True, True]:
        return "B"

    # C
    if (
        not index and not middle and not ring and not pinky
        and 0.08 < thumb_index_dist < 0.25
    ):
        return "C"

    # D
    if index and not middle and not ring and not pinky:
        return "D"

    # E
    if fingers == [False, False, False, False, False] and thumb_index_dist < 0.12:
        return "E"

    # F
    if thumb_index_dist < 0.06 and middle and ring and pinky:
        return "F"

    # G
    if index and not middle and not ring and not pinky and thumb:
        if abs(index_tip.y - index_mcp.y) < 0.12:
            return "G"

    # H
    if index and middle and not ring and not pinky:
        if index_middle_dist < 0.08 and abs(index_tip.y - index_mcp.y) < 0.15:
            return "H"

    # I
    if not index and not middle and not ring and pinky:
        return "I"

    # J is handled by MotionTracker

    # K
    if index and middle and not ring and not pinky and thumb:
        if thumb_middle_dist < 0.12:
            return "K"

    # L
    if thumb and index and not middle and not ring and not pinky:
        return "L"

    # M
    if fingers == [False, False, False, False, False]:
        if thumb_tip.x > index_mcp.x and thumb_tip.x < pinky_mcp.x:
            return "M"

    # N
    if fingers == [False, False, False, False, False]:
        if thumb_tip.x > index_mcp.x and thumb_tip.x < ring_mcp.x:
            return "N"

    # O
    if (
        thumb_index_dist < 0.08
        and thumb_middle_dist < 0.12
        and thumb_ring_dist < 0.14
        and thumb_pinky_dist < 0.16
    ):
        return "O"

    # P
    if index and middle and not ring and not pinky and thumb:
        if index_tip.y > wrist.y:
            return "P"

    # Q
    if index and not middle and not ring and not pinky and thumb:
        if index_tip.y > wrist.y:
            return "Q"

    # R
    if index and middle and not ring and not pinky:
        if index_middle_dist < 0.04:
            return "R"

    # S
    if fingers == [False, False, False, False, False]:
        return "S"

    # T
    if fingers == [False, False, False, False, False]:
        if distance(thumb_tip, index_mcp) < 0.10:
            return "T"

    # U
    if index and middle and not ring and not pinky:
        if index_middle_dist < 0.08:
            return "U"

    # V
    if index and middle and not ring and not pinky:
        if index_middle_dist >= 0.08:
            return "V"

    # W
    if not thumb and index and middle and ring and not pinky:
        return "W"

    # X
    if not thumb and not middle and not ring and not pinky:
        if lm[8].y > lm[6].y and lm[6].y < lm[5].y:
            return "X"

    # Y
    if thumb and not index and not middle and not ring and pinky:
        return "Y"

    # Z is handled by MotionTracker

    return "Unknown"


def draw_text_box(frame, text):
    """
    Bottom black text box.
    Shows current translation.
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


def draw_sentence_box(frame, sentence, status):
    """
    Top black text box.
    Shows full sentence being built.
    """
    height, width, _ = frame.shape

    cv2.rectangle(
        frame,
        (20, 20),
        (width - 20, 120),
        (0, 0, 0),
        -1
    )

    display_sentence = sentence if sentence != "" else "(empty)"

    cv2.putText(
        frame,
        f"Sentence: {display_sentence}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        status,
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA
    )


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    stable_prediction = StablePrediction(max_history=15, min_confidence_count=8)
    motion_tracker = MotionTracker(max_points=25)
    sentence_builder = SentenceBuilder(hold_time=1.0, cooldown=1.2)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("ASL alphabet translator started.")
    print("Hold a letter steady to lock it in.")
    print("Two open hands = space.")
    print("Two fists = clear sentence.")
    print("Backspace key = delete last character.")
    print("C key = clear sentence.")
    print("Q key = quit.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
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
                hand_data = []

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):
                        handedness_label = handedness.classification[0].label
                        hand_data.append((hand_landmarks, handedness_label))

                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

                    two_hand_command = detect_two_hand_command(hand_data)

                    if two_hand_command is not None:
                        current_prediction = two_hand_command
                        motion_tracker.clear()

                    else:
                        hand_landmarks, handedness_label = hand_data[0]

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

                else:
                    current_prediction = "No Hand Detected"
                    motion_tracker.clear()

                final_translation = stable_prediction.update(current_prediction)

                sentence_builder.update(final_translation)

                draw_sentence_box(
                    frame,
                    sentence_builder.text,
                    sentence_builder.status
                )

                draw_text_box(frame, final_translation)

                cv2.putText(
                    frame,
                    "Hold letter to type | Two open hands = Space | Two fists = Clear | Backspace = Delete | Q = Quit",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("ASL Alphabet Translator", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("c"):
                    sentence_builder.clear()
                    stable_prediction.clear()
                    motion_tracker.clear()

                elif key == 8 or key == 127:
                    sentence_builder.backspace()

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

                sentence_builder.update(final_translation)

                draw_sentence_box(
                    frame,
                    sentence_builder.text,
                    sentence_builder.status
                )

                draw_text_box(frame, final_translation)

                cv2.putText(
                    frame,
                    "Hold letter to type | B/Open Hand = Space | S/Fist = Clear | Backspace = Delete | Q = Quit",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("ASL Alphabet Translator", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("c"):
                    sentence_builder.clear()
                    stable_prediction.clear()
                    motion_tracker.clear()

                elif key == 8 or key == 127:
                    sentence_builder.backspace()

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
