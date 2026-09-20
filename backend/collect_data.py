import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request

# ============================================================
# 1. PRE-PLANNED DATASET CONFIGURATION
# ============================================================
ACTIONS = np.array(['namaste', 'thank_you', 'please'])
NO_SEQUENCES = 30     # Number of videos per sign
SEQUENCE_LENGTH = 30  # Frames per video

DATA_PATH = os.path.join('dataset', 'raw')

for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Auto-download MediaPipe Hand Landmarker model
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        MODEL_PATH
    )

# ============================================================
# 2. MEDIAPIPE TASKS INITIALIZATION
# ============================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# 3. RECORDING EXECUTION LOOP
# ============================================================
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    for action in ACTIONS:
        print(f"\n--- SCRIPTING CHANGE: PREPARE FOR '{action.upper()}' ---")
        time.sleep(3)
        
        for sequence in range(NO_SEQUENCES):
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp_ms = int(time.time() * 1000)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                results = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                display_frame = frame.copy()
                if frame_num == 0: 
                    cv2.putText(display_frame, 'STARTING COLLECTION', (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', display_frame)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(display_frame, f'Collecting {action} [{sequence}] f:{frame_num}', (15, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', display_frame)
                
                # Extract 21 landmarks * 3 coords (63 per hand) -> left/right mapping
                lh = np.zeros(21*3, dtype=np.float32)
                rh = np.zeros(21*3, dtype=np.float32)
                
                if results.hand_landmarks and results.handedness:
                    for idx, hand_lms in enumerate(results.hand_landmarks):
                        h_label = results.handedness[idx][0].display_name
                        coords = np.array([[res.x, res.y, res.z] for res in hand_lms], dtype=np.float32).flatten()
                        if h_label == 'Left':
                            lh = coords
                        else:
                            rh = coords
                
                # Combined 126 coordinate array for hand-focused ISL sign tracking
                keypoints = np.concatenate([lh, rh])
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()