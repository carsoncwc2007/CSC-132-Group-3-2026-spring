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

# Start PWM BUT force stop immediately
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

# ---------------- STARTUP WARMUP ----------------
print("[INFO] Camera warming up...")
start_time = time.time()

while time.time() - start_time < 2:  # 2 second warmup
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        cv2.imshow("Warming Up Camera...", frame)
        cv2.waitKey(1)

print("[INFO] Camera ready.")

# Now ensure servos are STOPPED after warmup
stop_all()
time.sleep(1)  # extra stabilization

# ---------------- TRACKING VARIABLES ----------------
prev_x = None
prev_y = None

MOVE_THRESHOLD = 20
MOVE_DELAY = 0.6   # delay before reacting to movement
last_face_time = time.time()

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

            current_time = time.time()

            # Only react AFTER a short delay (prevents instant twitch)
            if current_time - last_face_time > MOVE_DELAY:

                if prev_x is not None and prev_y is not None:
                    dx = face_x - prev_x
                    dy = face_y - prev_y

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

            last_face_time = current_time
            prev_x = face_x
            prev_y = face_y

        else:
            stop_all()
            prev_x = None
            prev_y = None
            last_face_time = time.time()

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
