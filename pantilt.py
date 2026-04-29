import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# ---------------- GPIO SETUP ----------------
GPIO.setmode(GPIO.BCM)

PAN_PIN = 6
TILT_PIN = 16

GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan_pwm = GPIO.PWM(PAN_PIN, 50)
tilt_pwm = GPIO.PWM(TILT_PIN, 50)

NEUTRAL = 7.5

# Start safely (no movement)
pan_pwm.start(0)
tilt_pwm.start(0)

def stop_all():
    pan_pwm.ChangeDutyCycle(NEUTRAL)
    tilt_pwm.ChangeDutyCycle(NEUTRAL)

def move_left():
    pan_pwm.ChangeDutyCycle(6.9)

def move_right():
    pan_pwm.ChangeDutyCycle(8.1)

def move_up():
    tilt_pwm.ChangeDutyCycle(8.1)

def move_down():
    tilt_pwm.ChangeDutyCycle(6.9)

# ---------------- MEDIAPIPE SETUP ----------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------- CAMERA WARMUP ----------------
print("[INFO] Warming up camera...")
start_time = time.time()

while time.time() - start_time < 2:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        cv2.imshow("Warming Up...", frame)
        cv2.waitKey(1)

print("[INFO] Camera ready")

# Ensure servos are stopped
stop_all()
time.sleep(1)

# ---------------- TRACKING VARIABLES ----------------
prev_x = None
prev_y = None

MOVE_THRESHOLD = 0.05   # normalized (MediaPipe uses 0–1 coords)
MOVE_DELAY = 0.6

last_move_time = time.time()

# ---------------- MAIN LOOP ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            # Convert normalized coords to pixels
            x = int(bbox.xmin * FRAME_WIDTH)
            y = int(bbox.ymin * FRAME_HEIGHT)
            w = int(bbox.width * FRAME_WIDTH)
            h = int(bbox.height * FRAME_HEIGHT)

            face_x = x + w // 2
            face_y = y + h // 2

            norm_x = face_x / FRAME_WIDTH
            norm_y = face_y / FRAME_HEIGHT

            # Draw
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(frame, (face_x, face_y), 5, (0,0,255), -1)

            current_time = time.time()

            if current_time - last_move_time > MOVE_DELAY:

                if prev_x is not None and prev_y is not None:
                    dx = norm_x - prev_x
                    dy = norm_y - prev_y

                    # -------- PAN --------
                    if dx > MOVE_THRESHOLD:
                        move_right()
                        time.sleep(0.04)
                        stop_all()

                    elif dx < -MOVE_THRESHOLD:
                        move_left()
                        time.sleep(0.04)
                        stop_all()

                    # -------- TILT --------
                    if dy > MOVE_THRESHOLD:
                        move_down()
                        time.sleep(0.04)
                        stop_all()

                    elif dy < -MOVE_THRESHOLD:
                        move_up()
                        time.sleep(0.04)
                        stop_all()

                last_move_time = current_time

            prev_x = norm_x
            prev_y = norm_y

        else:
            stop_all()
            prev_x = None
            prev_y = None
            last_move_time = time.time()

        cv2.imshow("MediaPipe Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# ---------------- CLEANUP ----------------
finally:
    cap.release()
    cv2.destroyAllWindows()
    pan_pwm.stop()
    tilt_pwm.stop()
    GPIO.cleanup()
