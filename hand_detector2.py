import cv2
import mediapipe as mp

class handDetector():
    

    def __init__(self, mode=False, max_hands=2, detection_con=.5, presence_con=.5, track_con=.5) -> None:
        
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.presence_con = presence_con
        self.track_con = track_con
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw =  mp.solutions.drawing_utils


    def find_hands(self, img, draw=True):
        

        #Turns the image into an RGB image and checks for hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        #Draws hand landmarks if draw is True and there are hands detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img
    
    def find_position(self, img, draw=True):
        

        #Initialize lists for all the landmarks
        all_landmarks = []
        
        #Checks for hands in the image
        if self.results.multi_hand_landmarks and self.results.multi_handedness:
            
            #Loops through each detected hand
            for hand_num in range(len(self.results.multi_handedness)):

                #Gathers the handedness for of each detected hand
                hand = self.results.multi_hand_landmarks[hand_num]
                handedness = self.results.multi_handedness[hand_num].classification

                #Loops through each classification for each hand
                for classification in handedness:
                    
                    #Initialize individual list for hand landmarks
                    landmark_list = []

                    #Enumerates through each id and landmark
                    for id, landmark in enumerate(hand.landmark):

                        #Calculates the coordinates of each landmark in comparison to the size of the image window
                        height, width, center = img.shape
                        center_x, center_y = int(landmark.x*width), int(landmark.y*height)
                        
                        #Adds the id and coordinates to the landmark list
                        landmark_list.append([id, center_x, center_y])
                        
                        #Optionally draws the location of each landmark
                        if draw:
                            cv2.circle(img, (center_x,center_y), 5, (255,255,255), cv2.FILLED)
                    
                    #Adds the handedness and landmark list to the larger list of landmarks
                    all_landmarks.append((classification.label, landmark_list))
        
        return all_landmarks