def normalize_landmarks(hand_landmarks, hand_index=0):
    """
    INPUT:
      hand_landmarks: MediaPipe hand_landmarks object (21 points × 3D coordinates)
      hand_index: which hand (0=right, 1=left) — not used here, but placeholder
    
    OUTPUT:
      numpy array of shape (63,) containing normalized [x1,y1,z1,x2,y2,z2,...,x21,y21,z21]
    
    ALGORITHM:
      1. Extract raw coordinates from MediaPipe landmarks
         - Each landmark has .x, .y, .z (z = depth), ranges roughly [0, 1]
      
      2. Center on wrist (landmark index 0)
         - Wrist is the reference point (index 0 in MediaPipe hand)
         - Subtract wrist coordinates from all others
         - Result: hand is centered at origin
      
      3. Scale by hand bounding box
         - Find max/min x, y across all landmarks
         - Compute width = max_x - min_x, height = max_y - min_y
         - Divide all normalized coordinates by max(width, height)
         - Result: hand fits in ~[-0.5, 0.5] box regardless of size
      
      4. Flatten to 1D array [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
    
    WHY NORMALIZE:
      - Position-invariant: Hand position in frame doesn't matter
      - Scale-invariant: Hand distance from camera doesn't matter (as much)
      - Rotation: Less critical for fingerspelling (hand orientation varies)
        but could add rotation alignment if needed
    """
    # Normalize the landmarks to a range of [0, 1]
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist_landmark = landmarks_array[0]  # Wrist landmark
    centered=landmarks_array - wrist_landmark  # Center the landmarks around the wrist
    xs,ys,zs=centered[:,0],centered[:,1],centered[:,2]
    width=np.max(xs)-np.min(xs)
    height=np.max(ys)-np.min(ys)
    scale = max(width, height,0.0001)  # Avoid division by zero
    scaled=centered/scale
    flattened=scaled.flatten()
    return flattened

def extract_single_hand(results, hand_choice="dominant"):
    """
    INPUT:
      results: MediaPipe detection results (contains multi_hand_landmarks)
      hand_choice: 'dominant', 'left', 'right', or 0 (first detected)
    
    OUTPUT:
      Single hand_landmarks object or None if not detected
    
    RATIONALE:
      - MediaPipe can detect both hands
      - For fingerspelling, typically use one hand at a time
      - This function picks the right one
    """
    if not results.multi_hand_landmarks:
        return None
    if hand_choice=="dominant" or hand_choice==0:
        return results.multi_hand_landmarks[0]
    elif hand_choice=="left" and len(results.multi_hand_landmarks) > 1:
        return results.multi_hand_landmarks[1]
    elif hand_choice=="right":
        return results.multi_hand_landmarks[0]
    else:
        return results.multi_hand_landmarks[0]  

def get_detection_confidence(results):
    """
    INPUT:
      results: MediaPipe detection results
    
    OUTPUT:
      float in [0, 1] — average confidence of all detected hands
    
    USE CASE:
      - Filter out unreliable detections
      - Only predict letter if confidence > 0.6 (hand clearly visible)
    """
    if not results.muti_hand_landmarks or not results.multi_handedness:
        return 0.0
    confidences= [hand.classificaiton[0].score
                    for hand in results.multi_handedness]
    return np.mean(confidences)

def is_valid_feature_vector(feature_vector, min_length=63):
    """
    Sanity check before sending to classifier
    - Ensure correct shape
    - No NaN/Inf values
    - Magnitude not too large (indicates bad normalization)
    """
    if feature_vector is None or len(feature_vector) < min_length:
        return False
    if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
        return False
    if np.max(np.abs(feature_vector))>10:
        return False
    return True
