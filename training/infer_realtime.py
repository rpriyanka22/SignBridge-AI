import cv2
import joblib
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from scipy.stats import mode
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

# ---------------------------------------------------------
# 1. Matching PyTorch LSTM Architecture (Hidden Size = 32, Mean Pooling)
# ---------------------------------------------------------
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)  # Mean pooling across frames
        out = self.fc(out)
        return out

print("Loading model and preprocessors...")
scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
label_encoder = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
classes = label_encoder.classes_

# Initialize Model
input_size = scaler.mean_.shape[0]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SignLSTM(input_size=input_size, hidden_size=32, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(PROCESSED_DIR / "lstm_model.pth", map_location=device))
model.eval()

# ---------------------------------------------------------
# 2. Buffers & Thresholds
# ---------------------------------------------------------
SEQ_LEN = 30
CONFIDENCE_THRESHOLD = 0.70  # Require 70% probability to output a sign
sequence_buffer = deque(maxlen=SEQ_LEN)
prediction_buffer = deque(maxlen=10)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_features(results):
    """
    REPLACE THIS with your exact static feature extraction logic.
    """
    # Example:
    # pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    # return pose
    pass

def main():
    cap = cv2.VideoCapture(0)
    current_prediction = "Waiting..."
    confidence_display = 0.0
    previous_features = None

    with holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            try:
                current_features = extract_features(results)
                
                if current_features is not None and np.sum(current_features) != 0:
                    # Calculate frame-to-frame delta
                    if previous_features is None:
                        deltas = np.zeros_like(current_features)
                    else:
                        deltas = current_features - previous_features
                    previous_features = current_features
                    
                    # Scale static + motion features
                    combined_features = np.concatenate([current_features, deltas])
                    features_scaled = scaler.transform(combined_features.reshape(1, -1))[0]
                    
                    sequence_buffer.append(features_scaled)
                    
                    # Evaluate when sequence buffer is full
                    if len(sequence_buffer) == SEQ_LEN:
                        seq_tensor = torch.FloatTensor(np.array(sequence_buffer)).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            logits = model(seq_tensor)
                            probabilities = F.softmax(logits, dim=1)
                            max_prob, predicted = torch.max(probabilities, 1)
                            
                            prob_val = max_prob.item()
                            pred_idx = predicted.item()

                        # Confidence Threshold Check
                        if prob_val >= CONFIDENCE_THRESHOLD:
                            prediction_buffer.append(pred_idx)
                            majority_vote = mode(prediction_buffer, keepdims=False)[0]
                            current_prediction = classes[majority_vote]
                            confidence_display = prob_val
                        else:
                            current_prediction = "Idle / Uncertain"
                            confidence_display = prob_val
                else:
                    previous_features = None
                    sequence_buffer.clear()
                    prediction_buffer.clear()
                    current_prediction = "No sign detected"
                    confidence_display = 0.0

            except Exception as e:
                pass

            # UI Display
            cv2.rectangle(image_bgr, (0, 0), (640, 50), (245, 117, 16), -1)
            status_text = f"SIGN: {current_prediction} ({confidence_display*100:.1f}%)" if confidence_display > 0 else f"SIGN: {current_prediction}"
            cv2.putText(image_bgr, status_text, (15, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('SignBridge AI - 95% LSTM Inference', image_bgr)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()