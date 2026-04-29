import cv2
import RPi.GPIO as GPIO
import time
import os

# ---------------- GPIO SETUP ----------------
GPIO.setmode(GPIO.BCM)

PAN_PIN = 6
TILT_PIN = 16

GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan_pwm = GPIO.PWM(PAN_PIN, 50)
tilt_pwm = GPIO.PWM(TILT_PIN, 50)

NEUTRAL = 7.5
pan_pwm.start(NEUTRAL)
tilt_pwm.start(NEUTRAL)

# ---------------- SERVO CONTROL ----------------
def stop_all():
    pan_pwm.ChangeDutyCycle(NEUTRAL)
    tilt_pwm.ChangeDutyCycle(NEUTRAL)

def move_left():
    pan_pwm.ChangeDutyCycle(6.8)

def move_right():
    pan_pwm.ChangeDutyCycle(8.2)

def move_up():
    tilt_pwm.ChangeDutyCycle(8.2)

def move_down():
    tilt_pwm.ChangeDutyCycle(6.8)

# ---------------- FIND CASCADE ----------------
def get_cascade_path():
    paths = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise Exception("Cascade file not found")

face_cascade = cv2.CascadeClassifier(get_cascade_path())

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------- TRACKING VARIABLES ----------------
prev_x = None
prev_y = None

MOVE_THRESHOLD = 25     # how much movement triggers servo
COOLDOWN = 0.15        # seconds between moves
last_move_time = 0

# ---------------- MAIN LOOP ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]

            face_x = x + w // 2
            face_y = y + h // 2

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(frame, (face_x, face_y), 5, (0,0,255), -1)

            if prev_x is not None and prev_y is not None:
                dx = face_x - prev_x
                dy = face_y - prev_y

                current_time = time.time()

                # Only move if enough time passed (prevents jitter)
                if current_time - last_move_time > COOLDOWN:

                    # -------- PAN (LEFT/RIGHT) --------
                    if dx > MOVE_THRESHOLD:
                        move_right()
                        time.sleep(0.05)
                        pan_pwm.ChangeDutyCycle(NEUTRAL)

                    elif dx < -MOVE_THRESHOLD:
                        move_left()
                        time.sleep(0.05)
                        pan_pwm.ChangeDutyCycle(NEUTRAL)

                    # -------- TILT (UP/DOWN) --------
                    if dy > MOVE_THRESHOLD:
                        move_down()
                        time.sleep(0.05)
                        tilt_pwm.ChangeDutyCycle(NEUTRAL)

                    elif dy < -MOVE_THRESHOLD:
                        move_up()
                        time.sleep(0.05)
                        tilt_pwm.ChangeDutyCycle(NEUTRAL)

                    last_move_time = current_time

            # Update previous position
            prev_x = face_x
            prev_y = face_y

        else:
            stop_all()
            prev_x = None
            prev_y = None

        cv2.imshow("Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# ---------------- CLEANUP ----------------
finally:
    cap.release()
    cv2.destroyAllWindows()
    pan_pwm.stop()
    tilt_pwm.stop()
    GPIO.cleanup()
