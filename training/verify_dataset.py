from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = Path("../dataset/processed/landmarks.csv")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SIGNBRIDGE AI - DATASET VERIFICATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not DATASET_FILE.exists():

        print("\nERROR: landmarks.csv was not found.")

        print(
            "\nExpected location:"
        )

        print(
            DATASET_FILE.resolve()
        )

        return

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET_FILE)

    print("Dataset loaded successfully.")

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("BASIC INFORMATION")
    print("-" * 60)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Expected landmark columns
    # --------------------------------------------------------

    landmark_columns = []

    for i in range(21):

        landmark_columns.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}",
        ])

    missing_columns = [
        column
        for column in landmark_columns
        if column not in df.columns
    ]

    print(
        f"\nExpected landmark features: "
        f"{len(landmark_columns)}"
    )

    if missing_columns:

        print("\nERROR: Missing landmark columns:")

        for column in missing_columns:
            print(f"    {column}")

    else:

        print(
            "All 63 landmark features are present."
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("SIGN LABELS")
    print("-" * 60)

    labels = sorted(
        df["label"].unique()
    )

    print(
        f"\nNumber of signs: {len(labels)}"
    )

    for label in labels:

        count = (
            df["label"] == label
        ).sum()

        print(
            f"    {label}: {count}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("MISSING VALUES")
    print("-" * 60)

    total_missing = int(
        df.isnull().sum().sum()
    )

    print(
        f"\nTotal missing values: "
        f"{total_missing}"
    )

    if total_missing == 0:

        print(
            "No missing values found."
        )

    else:

        print(
            "\nColumns containing missing values:"
        )

        missing = (
            df.isnull()
            .sum()
        )

        print(
            missing[
                missing > 0
            ].to_string()
        )

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("DUPLICATES")
    print("-" * 60)

    duplicate_count = int(
        df.duplicated().sum()
    )

    print(
        f"\nDuplicate rows: "
        f"{duplicate_count}"
    )

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("DATA TYPES")
    print("-" * 60)

    print(
        df.dtypes.value_counts()
    )

    # --------------------------------------------------------
    # Coordinate range
    # --------------------------------------------------------

    coordinate_columns = landmark_columns

    print("\n" + "-" * 60)
    print("LANDMARK RANGE CHECK")
    print("-" * 60)

    coordinate_data = df[
        coordinate_columns
    ]

    print(
        f"\nMinimum coordinate value: "
        f"{coordinate_data.min().min():.4f}"
    )

    print(
        f"Maximum coordinate value: "
        f"{coordinate_data.max().max():.4f}"
    )

    # --------------------------------------------------------
    # Dataset preview
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("FIRST 5 ROWS")
    print("-" * 60)

    print(
        df.head().to_string()
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

    if (
        not missing_columns
        and total_missing == 0
    ):

        print(
            "\nDataset looks ready for the next step."
        )

    else:

        print(
            "\nDataset needs attention before training."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()