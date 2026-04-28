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

pan_pwm = GPIO.PWM(PAN_PIN, 50)   # 50Hz
tilt_pwm = GPIO.PWM(TILT_PIN, 50)

NEUTRAL = 7.5

pan_pwm.start(NEUTRAL)
tilt_pwm.start(NEUTRAL)

# ---------------- SERVO CONTROL ----------------
def stop_servo(pwm):
    pwm.ChangeDutyCycle(NEUTRAL)

def move_left():
    pan_pwm.ChangeDutyCycle(6.8)

def move_right():
    pan_pwm.ChangeDutyCycle(8.2)

def move_up():
    tilt_pwm.ChangeDutyCycle(8.2)

def move_down():
    tilt_pwm.ChangeDutyCycle(6.8)

# ---------------- FIND CASCADE FILE ----------------
def get_cascade_path():
    possible_paths = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"[INFO] Using cascade file: {path}")
            return path

    raise Exception("Haar cascade file not found. Install OpenCV properly.")

cascade_path = get_cascade_path()
face_cascade = cv2.CascadeClassifier(cascade_path)

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FRAME_CENTER_X = FRAME_WIDTH // 2
FRAME_CENTER_Y = FRAME_HEIGHT // 2

TOLERANCE = 40  # dead zone to prevent jitter

# ---------------- MAIN LOOP ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # track first face

            face_x = x + w // 2
            face_y = y + h // 2

            # Draw visuals
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (FRAME_CENTER_X, FRAME_CENTER_Y), 5, (255, 0, 0), -1)

            # -------- PAN CONTROL --------
            if face_x < FRAME_CENTER_X - TOLERANCE:
                move_left()
            elif face_x > FRAME_CENTER_X + TOLERANCE:
                move_right()
            else:
                stop_servo(pan_pwm)

            # -------- TILT CONTROL --------
            if face_y < FRAME_CENTER_Y - TOLERANCE:
                move_up()
            elif face_y > FRAME_CENTER_Y + TOLERANCE:
                move_down()
            else:
                stop_servo(tilt_pwm)

        else:
            # No face detected → stop movement
            stop_servo(pan_pwm)
            stop_servo(tilt_pwm)

        cv2.imshow("Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

        time.sleep(0.02)  # smooth movement

# ---------------- CLEANUP ----------------
finally:
    cap.release()
    cv2.destroyAllWindows()
    pan_pwm.stop()
    tilt_pwm.stop()
    GPIO.cleanup()
