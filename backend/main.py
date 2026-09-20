import json
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "dataset" / "processed"

MODEL_FILE = MODEL_DIR / "lstm_model.pth"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"


# ============================================================
# PYTORCH LSTM ARCHITECTURE
# ============================================================

class SignLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out


# ============================================================
# LOAD AI COMPONENTS
# ============================================================

print("=" * 60)
print("SIGNBRIDGE AI - HOLISTIC CONTINUOUS BACKEND v3.1.0")
print("=" * 60)

has_pytorch_model = (
    MODEL_FILE.exists() and SCALER_FILE.exists() and LABEL_ENCODER_FILE.exists()
)

if has_pytorch_model:
    try:
        scaler = joblib.load(SCALER_FILE)
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
        classes = label_encoder.classes_

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = scaler.mean_.shape[0]
        num_classes = len(classes)

        model = SignLSTM(
            input_size=input_size, hidden_size=32, num_classes=num_classes
        ).to(device)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()

        print(f"Device: {device}")
        print(f"Number of input features: {input_size}")
        print(f"Number of target classes: {num_classes}")
        print("PyTorch LSTM model loaded successfully.")
    except Exception as e:
        has_pytorch_model = False
        print(f"Failed to load PyTorch model: {e}")
        print("Falling back exclusively to High-Precision Geometric Engine.")
else:
    print("PyTorch model files missing. Engine running in High-Precision Geometric Mode.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SignBridge AI API",
    description="Continuous Sentence & Holistic Multi-modal ISL recognition backend.",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST SCHEMAS & DATA EXTRACTION
# ============================================================

class LandmarkFrameRequest(BaseModel):
    left_hand: Optional[List[Dict[str, float]]] = []
    right_hand: Optional[List[Dict[str, float]]] = []
    face: Optional[List[Dict[str, float]]] = []
    pose: Optional[List[Dict[str, float]]] = []


def extract_holistic_keypoints(data: dict) -> np.ndarray:
    lh = np.array([[res['x'], res['y'], res.get('z', 0.0)] for res in data.get('left_hand', [])]).flatten() if data.get('left_hand') else np.zeros(21 * 3)
    rh = np.array([[res['x'], res['y'], res.get('z', 0.0)] for res in data.get('right_hand', [])]).flatten() if data.get('right_hand') else np.zeros(21 * 3)
    return np.concatenate([lh, rh]).astype(np.float32)


# ============================================================
# GEOMETRIC HEURISTIC ENGINE (15 SCALE-INVARIANT GESTURES)
# ============================================================

def euclidean_distance(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2
        + (p1["y"] - p2["y"]) ** 2
        + (p1.get("z", 0.0) - p2.get("z", 0.0)) ** 2
    )


def parse_hand_landmarks(raw_hands: List[Any]) -> Optional[List[Dict[str, float]]]:
    if not raw_hands or len(raw_hands) == 0:
        return None

    if isinstance(raw_hands[0], dict):
        return raw_hands[:21]

    n = len(raw_hands)
    pts = []
    if n >= 63:
        for i in range(0, 63, 3):
            pts.append({
                "x": float(raw_hands[i]),
                "y": float(raw_hands[i + 1]),
                "z": float(raw_hands[i + 2]),
            })
    elif n >= 42:
        for i in range(0, 42, 2):
            pts.append({
                "x": float(raw_hands[i]),
                "y": float(raw_hands[i + 1]),
                "z": 0.0,
            })
    return pts if len(pts) == 21 else None


def classify_static_gesture(raw_hands: List[Any]) -> Tuple[Optional[str], float]:
    pts = parse_hand_landmarks(raw_hands)
    if not pts:
        return None, 0.0

    # CORRECTED INDICES FOR MEDIAPIPE LANDMARKS
    wrist = pts[0]
    thumb_tip, thumb_mcp = pts[4], pts[2]
    index_tip, index_pip = pts[8], pts[6]
    middle_tip, middle_pip = pts[12], pts[10]
    ring_tip, ring_pip = pts[16], pts[14]
    pinky_tip, pinky_pip, pinky_mcp = pts[20], pts[18], pts[17]

    hand_scale = euclidean_distance(wrist, index_pip) or 1.0

    index_ext = euclidean_distance(index_tip, wrist) > euclidean_distance(index_pip, wrist)
    middle_ext = euclidean_distance(middle_tip, wrist) > euclidean_distance(middle_pip, wrist)
    ring_ext = euclidean_distance(ring_tip, wrist) > euclidean_distance(ring_pip, wrist)
    pinky_ext = euclidean_distance(pinky_tip, wrist) > euclidean_distance(pinky_pip, wrist)
    thumb_ext = euclidean_distance(thumb_tip, pinky_mcp) > euclidean_distance(thumb_mcp, pinky_mcp)

    thumb_index_dist = euclidean_distance(thumb_tip, index_tip) / hand_scale
    index_middle_dist = euclidean_distance(index_tip, middle_tip) / hand_scale
    thumb_middle_dist = euclidean_distance(thumb_tip, middle_tip) / hand_scale

    if thumb_index_dist < 0.45 and middle_ext and ring_ext and pinky_ext:
        return "Okay", 96.0
    if index_ext and middle_ext and not ring_ext and not pinky_ext and index_middle_dist < 0.35:
        return "Crossed Fingers", 94.0
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if thumb_tip["y"] < thumb_mcp["y"] and thumb_tip["y"] < index_pip["y"]:
            return "Thumbs Up", 98.0
        if thumb_tip["y"] > thumb_mcp["y"] and thumb_tip["y"] > index_pip["y"]:
            return "Thumbs Down", 98.0
    if not thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "Closed Fist", 95.0
    if thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
        return "Open Palm", 98.0
    if index_ext and middle_ext and not ring_ext and not pinky_ext and index_middle_dist >= 0.20:
        return "Peace", 96.0
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext and thumb_middle_dist >= 0.35:
        return "Index Point", 96.0
    if thumb_ext and index_ext and pinky_ext and not middle_ext and not ring_ext:
        return "I Love You", 97.0
    if index_ext and pinky_ext and not middle_ext and not ring_ext and not thumb_ext:
        return "Rock On", 96.0
    if index_ext and middle_ext and ring_ext and not pinky_ext and not thumb_ext:
        return "Three", 95.0
    if index_ext and middle_ext and ring_ext and pinky_ext and not thumb_ext:
        return "Four", 96.0
    if thumb_index_dist < 0.30 and not middle_ext and not ring_ext and not pinky_ext:
        return "Pinch", 94.0

    # --- NEW SIGNS ADDED ---
    if pinky_ext and not index_ext and not middle_ext and not ring_ext and not thumb_ext:
        return "Pinky Promise", 96.0
    if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
        return "Hang Loose", 95.0
    if index_ext and middle_ext and not ring_ext and not pinky_ext and index_middle_dist < 0.20:
        return "Two Pointing", 95.0
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and thumb_middle_dist < 0.35:
        return "Letter D", 94.0
    if thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext and index_middle_dist > 0.40:
        return "L Shape", 95.0

    return None, 0.0


# ============================================================
# SEMANTIC VOCABULARY & SMART SENTENCE CONSTRUCTOR
# ============================================================

SEMANTIC_TOKENS = {
    "Open Palm": "Hello",
    "Thumbs Up": "Yes",
    "Thumbs Down": "No",
    "Closed Fist": "Stop",
    "I Love You": "Love",
    "Call Me": "Call",
    "Peace": "Peace",
    "Okay": "Okay",
    "Index Point": "Look",
    "Pinch": "Little",
    "L Shape": "Guide",
    "Rock On": "Cool",
    "Three": "Three",
    "Four": "Four",
    "Crossed Fingers": "Hope",
    "Pinky Promise": "Promise",
    "Two Pointing": "Watch",
    "Letter D": "Where",
    "Hang Loose": "Relax",
}

def construct_smart_sentence(words: List[str]) -> str:
    if not words:
        return "Waiting for gestures..."

    # 1. Map gestures to semantic meaning
    mapped: List[str] = [SEMANTIC_TOKENS.get(w.strip(), w.strip()) for w in words if w.strip()]
    
    if not mapped:
        return "Waiting for gestures..."

    # 2. Filter out rapid duplicate toggle noise (e.g., Hello Hello Hello)
    filtered = []
    for word in mapped:
        if not filtered or filtered[-1].lower() != word.lower():
            filtered.append(word)

    phrase_lower = " ".join(filtered).lower()
    
    # 3. Smart Contextual Overrides (Pattern Matching)
    templates = {
        "hello love": "Hello, I love you!",
        "hello call": "Hello, please call me.",
        "yes okay": "Yes, everything is okay.",
        "no stop": "No, please stop immediately.",
        "hope okay": "I hope everything is okay.",
        "promise love": "I promise I love you.",
        "stop look": "Stop and look here.",
        "hello where": "Hello, where are you?",
    }
    
    for pattern, output in templates.items():
        if pattern in phrase_lower:
            return output

    # 4. Default grammar cleanup
    sentence = " ".join(filtered)
    sentence = sentence.capitalize()
    if not sentence.endswith((".", "!", "?")):
        sentence += "."

    return sentence


def align_feature_vector(raw_features: np.ndarray, target_size: int) -> np.ndarray:
    current_size = len(raw_features)
    if current_size < target_size:
        return np.pad(raw_features, (0, target_size - current_size), mode="constant")
    elif current_size > target_size:
        return raw_features[:target_size]
    return raw_features


# ============================================================
# REST ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "SignBridge AI Backend is online.",
        "status": "online",
        "engine": "Hybrid Geometric + PyTorch LSTM Engine",
    }


@app.post("/predict")
def predict_single_frame(request: LandmarkFrameRequest):
    active_hand = request.right_hand if request.right_hand else request.left_hand
    if not active_hand or len(active_hand) == 0:
        return {"detected_sign": "--", "confidence": 0.0}

    static_sign, static_conf = classify_static_gesture(active_hand)
    if static_sign:
        return {"detected_sign": static_sign, "confidence": static_conf}

    if has_pytorch_model:
        current_landmarks = extract_holistic_keypoints(request.dict())
        deltas = np.zeros_like(current_landmarks)
        combined = np.concatenate([current_landmarks, deltas])
        combined_features = align_feature_vector(combined, input_size)
        scaled = scaler.transform(combined_features.reshape(1, -1))[0]

        seq = np.tile(scaled, (30, 1))
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(seq_tensor)
            probs = F.softmax(logits, dim=1)
            max_prob, predicted = torch.max(probs, 1)

        prob_val = float(max_prob.item())
        pred_sign = str(classes[predicted.item()])

        return {
            "detected_sign": pred_sign if prob_val >= 0.50 else "--",
            "confidence": round(prob_val * 100, 1),
        }

    return {"detected_sign": "Idle", "confidence": 0.0}


# ============================================================
# WEBSOCKET REAL-TIME STREAMING ENDPOINT
# ============================================================

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected.")

    SEQ_LEN = 30
    CONFIDENCE_THRESHOLD = 0.70  # Increased to require stronger AI confidence for smoothness
    STABILITY_THRESHOLD = 8      # Increased to require 8 consecutive frames of the same sign

    sequence_buffer = deque(maxlen=SEQ_LEN)
    prediction_buffer = deque(maxlen=10)
    stability_buffer = deque(maxlen=STABILITY_THRESHOLD)

    sentence: List[str] = []
    last_added_sign = None
    previous_landmarks = None
    missed_frames = 0
    last_sign_time = time.time()

    try:
        while True:
            data = await websocket.receive_json()

            # ACTION HANDLER: CLEAR STREAM
            if data.get("action") == "clear":
                sentence.clear()
                last_added_sign = None
                sequence_buffer.clear()
                prediction_buffer.clear()
                stability_buffer.clear()
                await websocket.send_json({
                    "hand_detected": False,
                    "detected_sign": "--",
                    "confidence": 0.0,
                    "translated_text": "Waiting...",
                    "sentence": [],
                    "constructed_sentence": "Waiting for gestures...",
                    "is_new_sign": False,
                    "word_count": 0,
                })
                continue

            # ACTION HANDLER: REMOVE SINGLE CHIP
            if data.get("action") == "remove_index":
                idx = data.get("index")
                if idx is not None and isinstance(idx, int) and 0 <= idx < len(sentence):
                    sentence.pop(idx)
                await websocket.send_json({
                    "hand_detected": True,
                    "detected_sign": "--",
                    "confidence": 0.0,
                    "translated_text": "Waiting...",
                    "sentence": sentence,
                    "constructed_sentence": construct_smart_sentence(sentence),
                    "is_new_sign": False,
                    "word_count": len(sentence),
                })
                continue

            raw_lh = data.get("left_hand", [])
            raw_rh = data.get("right_hand", [])

            if not raw_lh and not raw_rh and not data.get("face") and not data.get("pose"):
                missed_frames += 1
                stability_buffer.clear()

                if missed_frames > 10:
                    sequence_buffer.clear()
                    prediction_buffer.clear()
                    previous_landmarks = None

                if time.time() - last_sign_time > 1.5:
                    last_added_sign = None

                await websocket.send_json({
                    "hand_detected": False,
                    "detected_sign": "--",
                    "confidence": 0.0,
                    "translated_text": "Waiting...",
                    "sentence": sentence,
                    "constructed_sentence": construct_smart_sentence(sentence),
                    "is_new_sign": False,
                    "word_count": len(sentence),
                })
                continue

            missed_frames = 0
            detected_sign = "Idle"
            confidence_pct = 0.0

            active_hand = raw_rh if raw_rh else raw_lh
            static_sign, static_conf = classify_static_gesture(active_hand)

            if static_sign:
                detected_sign = static_sign
                confidence_pct = static_conf
            elif has_pytorch_model:
                current_landmarks = extract_holistic_keypoints(data)
                deltas = np.zeros_like(current_landmarks) if previous_landmarks is None or len(previous_landmarks) != len(current_landmarks) else current_landmarks - previous_landmarks
                previous_landmarks = current_landmarks

                combined_features = np.concatenate([current_landmarks, deltas])
                aligned_features = align_feature_vector(combined_features, input_size)
                scaled_features = scaler.transform(aligned_features.reshape(1, -1))[0]
                sequence_buffer.append(scaled_features)

                if len(sequence_buffer) > 0:
                    seq_data = np.array(sequence_buffer)
                    if len(sequence_buffer) < SEQ_LEN:
                        padding = np.tile(seq_data[-1], (SEQ_LEN - len(sequence_buffer), 1))
                        seq_data = np.vstack([seq_data, padding])

                    seq_tensor = torch.FloatTensor(seq_data).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(seq_tensor)
                        probs = F.softmax(logits, dim=1)
                        max_prob, predicted = torch.max(probs, 1)

                    prob_val = max_prob.item()
                    pred_idx = predicted.item()

                    if prob_val >= CONFIDENCE_THRESHOLD:
                        prediction_buffer.append(pred_idx)
                        majority_vote = Counter(prediction_buffer).most_common(1)[0][0]
                        detected_sign = str(classes[majority_vote])
                        confidence_pct = round(prob_val * 100, 1)
                    else:
                        detected_sign = "Uncertain"
                        confidence_pct = round(prob_val * 100, 1)

            clean_sign = detected_sign.strip().lower()
            ignore_list = ["idle", "idle / uncertain", "uncertain", "waiting...", "waiting", "--", ""]

            is_new_sign = False

            if clean_sign not in ignore_list and confidence_pct >= 65.0:
                stability_buffer.append(detected_sign)

                if (
                    len(stability_buffer) == STABILITY_THRESHOLD
                    and len(set(stability_buffer)) == 1
                ):
                    stable_sign = stability_buffer[0]
                    # Debounce duplicate immediate repeats or rapid noise (increased to 2.5 for smoother flow)
                    if stable_sign != last_added_sign or (time.time() - last_sign_time > 2.5):
                        sentence.append(stable_sign)
                        last_added_sign = stable_sign
                        last_sign_time = time.time()
                        is_new_sign = True
                        stability_buffer.clear()
            else:
                stability_buffer.clear()

            # Map semantic token for readable translated preview
            semantic_preview = SEMANTIC_TOKENS.get(detected_sign, detected_sign) if confidence_pct >= 50 else "Waiting..."

            await websocket.send_json({
                "hand_detected": True,
                "detected_sign": detected_sign,
                "confidence": confidence_pct,
                "translated_text": semantic_preview,
                "sentence": sentence,
                "constructed_sentence": construct_smart_sentence(sentence),
                "is_new_sign": is_new_sign,
                "word_count": len(sentence),
            })

    except WebSocketDisconnect:
        print("WebSocket client disconnected clean.")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)