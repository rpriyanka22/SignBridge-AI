from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def find_csv():
    paths = [
        PROCESSED_DIR / "landmarks.csv",
        PROCESSED_DIR / "landmarks_features.csv",
        BASE_DIR / "landmarks.csv",
    ]
    for p in paths:
        if p.exists():
            return p
    matches = list(BASE_DIR.glob("**/*landmarks*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not locate any landmark CSV file.")

def main():
    print("=" * 60)
    print("SIGNBRIDGE AI - DATASET PREP (WITH MOTION FEATURES)")
    print("=" * 60)

    csv_path = find_csv()
    df = pd.read_csv(csv_path)

    meta_cols = {"video", "label", "frame"}
    feature_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # ---------------------------------------------------------
    # NEW: Engineer Temporal Motion Features (Deltas)
    # ---------------------------------------------------------
    print("Engineering temporal motion features (deltas)...")
    # Group by video and calculate the difference from the previous frame
    delta_df = df.groupby('video')[feature_cols].diff().fillna(0)
    
    # Rename columns to indicate they are velocities/deltas
    delta_df.columns = [f"{c}_delta" for c in feature_cols]
    
    # Concatenate original static features with the new motion features
    df = pd.concat([df, delta_df], axis=1)
    
    # Update feature_cols to include the new delta columns
    feature_cols = feature_cols + list(delta_df.columns)
    # ---------------------------------------------------------

    label_encoder = LabelEncoder()
    df["encoded_label"] = label_encoder.fit_transform(df["label"])

    video_df = df[["video", "encoded_label"]].drop_duplicates().reset_index(drop=True)

    # 5-Fold Stratified Group Split: Folds 1-4 (80%) -> Train | Fold 0 (20%) -> Test
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_v_idx, test_v_idx = next(sgkf.split(video_df, video_df["encoded_label"], groups=video_df["video"]))

    train_videos = set(video_df.iloc[train_v_idx]["video"])
    test_videos = set(video_df.iloc[test_v_idx]["video"])

    train_df = df[df["video"].isin(train_videos)].reset_index(drop=True)
    test_df = df[df["video"].isin(test_videos)].reset_index(drop=True)

    print("\n------------------------------------------------------------")
    print("DATASET SPLIT SUMMARY")
    print("------------------------------------------------------------")
    print(f"Training set: {len(train_df)} frames across {train_df['video'].nunique()} videos ({train_df['encoded_label'].nunique()} classes)")
    print(f"Test set:     {len(test_df)} frames across {test_df['video'].nunique()} videos ({test_df['encoded_label'].nunique()} classes)")
    print(f"Features:     {len(feature_cols)} (Static + Motion)")

    X_train_raw = train_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)

    np.save(PROCESSED_DIR / "y_train.npy", train_df["encoded_label"].values)
    np.save(PROCESSED_DIR / "y_test.npy", test_df["encoded_label"].values)

    train_df[["video", "encoded_label"]].to_csv(PROCESSED_DIR / "train_meta.csv", index=False)
    test_df[["video", "encoded_label"]].to_csv(PROCESSED_DIR / "test_meta.csv", index=False)

    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
    joblib.dump(label_encoder, PROCESSED_DIR / "label_encoder.pkl")
    print("\nPreparation complete! 80% Train / 20% Test Split Ready.")

if __name__ == "__main__":
    main()