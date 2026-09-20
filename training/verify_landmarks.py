from pathlib import Path
import pandas as pd

CSV_PATH = Path("../dataset/processed/landmarks.csv")

def verify():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH.resolve()} does not exist.")
        return

    df = pd.read_csv(CSV_PATH)
    print("=" * 60)
    print("SIGNBRIDGE AI - LANDMARK DATASET VERIFICATION")
    print("=" * 60)
    print(f"\nTotal Dataset Rows (Frames): {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    
    # Landmark coordinate columns (x0..z20)
    feature_cols = [c for c in df.columns if c.startswith(('x', 'y', 'z'))]
    print(f"Landmark Feature Columns: {len(feature_cols)} (Expected: 63)")
    
    # Check for missing values
    null_count = df.isnull().sum().sum()
    print(f"Missing (NaN) Values: {null_count}")

    print("\nRows Per Sign Class:")
    print("-" * 30)
    print(df['label'].value_counts().to_string())
    print("=" * 60)

if __name__ == "__main__":
    verify()