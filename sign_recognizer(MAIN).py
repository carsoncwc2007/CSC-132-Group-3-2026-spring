import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from hand_utils import normalize_landmarks, extract_single_hand, get_detection_confidence
from sign_model import SignLanguageModel

class SignRecognizer:
    def __init__(self,model_path="trained_models/sign_classifier.pkl",confidence_threshold=0.75,buffer_size=5):
        """Real-time sign language recognizer using webcam and trained model
        Args:
            model_path (str): Path to the trained model file
            confidence_threshold (float): Minimum confidence to accept prediction
            buffer_size (int): Number of recent predictions to average for stability
        """
        self.model = SignLanguageModel()
        self.model.load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.prediction_buffer=deque(maxlen=buffer_size)
        self.last_prediction = None
        
        #medaipipe setup
        self.mp_hands=mp.solutions.hands
        self.mp_drawing=mp.solutions.drawing_utils
        self.hands=self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    def predict_frame(self,frame):
        """Predict sign from a single video frame
        Args:frame (numpy array): BGR image from webcam
        Returns:
            predicted_letter (str): Predicted sign letter or None if confidence too low
            confidence (float): Confidence of the prediction
        """
        rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=self.hands.process(rgb_frame)
        predicted_letter=None
        confidence=0.0
        is_valid=False
        if results.multi_hand_landmarks:
            hand_landmarks=extract_single_hand(results,hand_choice="dominant")
            if hand_landmarks:
                #extract and normalize features
                feature_vector=normalize_landmarks(hand_landmarks)
                #run through model
                predicted_letter, confidence=self.model.predict(feature_vector)
                #check  if prediciton is valid
                if confidence > self.confidence_threshold:
                    is_valid=True
                    self.prediction_buffer.append(predicted_letter)
        #get smoothed prediction (majority vote in buffer)
        smoothed_letter =self.get_smoothed_prediction()
        
        return predicted_letter, confidence, is_valid, smoothed_letter
    def get_smoothed_prediction(self):
        """smooth predictions by taking majority vote in buffer
        Returns:
            smost common letter found in buffer
        """
        if not self.prediction_buffer:
            return None
        from collections import Counter
        counts=Counter(Self.prediction_buffer)
        most_common_letter, count=counts.most_common(1)[0][0]
        #only return if it has a strong majority (around 70% of buffer)
        consensus=counts[most_common_letter]/len(self.prediction_buffer)
        if consensus > 0.7:
            return most_common_letter
        else:
            return None
    def run_live(self):
        """runs sign recognizer from webcam
        args:
        max_frames:max frames to process(none for infinite)
        """
        cap=cv2.VideoCapture(0)
        frame_count=0
        recognized_text=[]
        print("\n"+ "="*50)
        print("Starting live sign recognition. Press 'q' to quit.")
        print("="*50)
        print("press 'space' to add space between words")
        print("="*50+"\n")
        
        while True:
            ret,frame=cap.read()
            if not ret:
                print("failed to grab frame ")
                break
            frame=cv2.flip(frame,1)
            frame_count+=1
            
            #predict
            predicted_letter, confidence, is_valid, smoothed_letter=self.predict_frame(frame)
            
            #update output if high confidence prediction 
            if smoothed and smoothed != self.last_prediction:
                recognized_text.append(smoothed)
                self.last_prediction=smoothed
                print(f"recognized: {smoothed} (confidence: {confidence:.2f})")
            #display recognized text on frame
            h,w,_=frame.shape
            #draw current prediction
            if pred_letter:
                color=(0,255,0) if is_valid else (0,0,255)
                cv2.putText(frame,f"Pred:{pred_letter} ({confidence:.2f})",
                            (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            #draw smoothed output
            if smoothed:
                cv2.putText(frame,f"Output:{smoothed}",
                            (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            current_text="".join(recognized_text)
            cv2.putText(frame,f"translation:{current_text}",
                        (10,120),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
            #draw bufffer info 
            cv2.putText(frame,f"Buffer:{len(self.prediction_buffer)}/{self.buffer_size}",
                        (10,16),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200,200,200),1)
            cv2.imshow("Sign Recognizer", frame)
            key=cv2.waitKey(1) & 0xFF
            if key== ord('q'):
                break
            elif key == ord(' '):
                recognized_text.append(" ")
                self.last_prediction=None
                print("added space")
            if max_frames and frame_count >= max_frames:
                print(f"Reached max frames ({max_frames}). Stopping.")
                break
        cap.release()
        cv2.destroyAllWindows()
        
        final_text="".join(recognized_text)
        print(f"\n"+"="*50)
        print(f"Final recognized text:{final_text}")
        print("="*50)
        return final_text
if __name__=="__main__":
    recognizer=SignRecognizer(
        model_path="trained_models/sign_classifier.pkl",
        confidence_threshold=0.75,
        buffer_size=5
    )
    recognizer.run_live()