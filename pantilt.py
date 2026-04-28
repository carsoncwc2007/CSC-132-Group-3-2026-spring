import cv2
import RPi.GPIO as GPIO
import time

# ---------------- GPIO SETUP ----------------
GPIO.setmode(GPIO.BCM)

PAN_PIN = 17
TILT_PIN = 27

GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan_pwm = GPIO.PWM(PAN_PIN, 50)   # 50Hz
tilt_pwm = GPIO.PWM(TILT_PIN, 50)

pan_pwm.start(7.5)   # neutral stop
tilt_pwm.start(7.5)

# ---------------- SERVO CONTROL ----------------
def stop_servo(pwm):
    pwm.ChangeDutyCycle(7.5)

def move_left():
    pan_pwm.ChangeDutyCycle(6.5)  # adjust for servo

def move_right():
    pan_pwm.ChangeDutyCycle(8.5)

def move_up():
    tilt_pwm.ChangeDutyCycle(8.5)

def move_down():
    tilt_pwm.ChangeDutyCycle(6.5)

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

FRAME_CENTER_X = 320
FRAME_CENTER_Y = 240
TOLERANCE = 40  # dead zone

# ---------------- MAIN LOOP ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_x = x + w // 2
            face_y = y + h // 2

            # Draw box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(frame, (face_x, face_y), 5, (0,0,255), -1)

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

            break  # track only first face

        cv2.imshow("Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pan_pwm.stop()
    tilt_pwm.stop()
    GPIO.cleanup()
