from pathlib import Path
from urllib.request import urlretrieve
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("../dataset/raw")
OUTPUT_DIR = Path("../dataset/processed")
OUTPUT_FILE = OUTPUT_DIR / "landmarks.csv"
MODEL_FILE = OUTPUT_DIR / "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FRAME_STEP = 5

# Key Landmark Indices
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
MIDDLE_MCP = 9

# ============================================================
# DOWNLOAD MEDIAPIPE MODEL
# ============================================================

def download_model():
    if MODEL_FILE.exists():
        print("MediaPipe hand model already exists.")
        return

    print("Downloading MediaPipe hand landmark model...")
    try:
        urlretrieve(MODEL_URL, MODEL_FILE)
    except Exception as error:
        print("\nERROR: Could not download MediaPipe model.", error)
        raise
    print("Model downloaded successfully.")

# ============================================================
# CREATE MEDIAPIPE HAND LANDMARKER
# ============================================================

def create_hand_landmarker():
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_FILE),
        delegate=mp.tasks.BaseOptions.Delegate.CPU,
    )
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)

# ============================================================
# FEATURE EXTRACTION (78 STATIC FEATURES)
# ============================================================

def extract_hand_landmarks(frame, landmarker, timestamp_ms):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not results.hand_landmarks:
        return None

    hand = results.hand_landmarks[0]
    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand])

    # 1. Wrist Normalization & Scale Centering
    wrist = raw_coords[WRIST]
    centered_coords = raw_coords - wrist
    hand_scale = np.linalg.norm(centered_coords[MIDDLE_MCP])
    
    if hand_scale > 0:
        centered_coords = centered_coords / hand_scale

    # 2. Compute Inter-Fingertip Pair Distances (10 features)
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    fingertip_distances = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dist = np.linalg.norm(centered_coords[tips[i]] - centered_coords[tips[j]])
            fingertip_distances.append(dist)

    # 3. Compute Tip-to-Wrist Distances (5 features)
    wrist_distances = [np.linalg.norm(centered_coords[tip]) for tip in tips]

    # Combine all 78 static features
    feature_vector = np.hstack([
        centered_coords.flatten(), # 63 normalized coords
        fingertip_distances,       # 10 tip pair dists
        wrist_distances            # 5 tip-wrist dists
    ])

    return feature_vector

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        print("Dataset folder not found:", DATASET_DIR.resolve())
        return

    print("=" * 60)
    print("SIGNBRIDGE AI - ENHANCED FEATURE EXTRACTION (156 FEATURES)")
    print("=" * 60)

    download_model()

    video_files = [
        file for file in DATASET_DIR.rglob("*")
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
    ]

    print(f"\nVideos found: {len(video_files)}")
    if not video_files:
        return

    landmarker = create_hand_landmarker()
    rows = []
    global_timestamp = 0

    try:
        for video_index, video_path in enumerate(video_files, start=1):
            relative_path = video_path.relative_to(DATASET_DIR)
            parts = relative_path.parts

            category = parts[0] if len(parts) >= 3 else "Unknown"
            label = parts[1] if len(parts) >= 3 else video_path.parent.name

            print(f"[{video_index}/{len(video_files)}] {label} -> {video_path.name}")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_number = 0
            prev_static_features = None

            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_number += 1
                if frame_number % FRAME_STEP != 0:
                    continue

                frame_timestamp = int((frame_number / fps) * 1000)
                timestamp_ms = global_timestamp + frame_timestamp
                if timestamp_ms <= global_timestamp:
                    timestamp_ms = global_timestamp + 1

                static_features = extract_hand_landmarks(frame, landmarker, timestamp_ms)
                global_timestamp = timestamp_ms

                if static_features is None:
                    continue

                # Velocity / Delta Features (78 values)
                if prev_static_features is None:
                    delta_features = np.zeros_like(static_features)
                else:
                    delta_features = static_features - prev_static_features
                
                prev_static_features = static_features.copy()

                row = {
                    "label": label,
                    "category": category,
                    "video": video_path.name,
                    "frame": frame_number,
                }

                # Save 63 Normalized Coordinates (x0..z20)
                for i in range(21):
                    row[f"x{i}"] = static_features[i * 3]
                    row[f"y{i}"] = static_features[i * 3 + 1]
                    row[f"z{i}"] = static_features[i * 3 + 2]

                # Save 15 Spatial Distance Features
                for i in range(10):
                    row[f"dist_pair_{i}"] = static_features[63 + i]
                for i in range(5):
                    row[f"dist_wrist_{i}"] = static_features[73 + i]

                # Save 78 Delta / Velocity Features
                for i in range(78):
                    row[f"delta_{i}"] = delta_features[i]

                rows.append(row)

            cap.release()
            global_timestamp += 1000

    finally:
        landmarker.close()

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"Rows generated: {len(dataframe)}")
    print("Total features per frame: 156 (78 Static Coords/Dists + 78 Motion Deltas)")
    print("=" * 60)

if __name__ == "__main__":
    main()