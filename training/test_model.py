from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

MODEL_FILE = (
    PROCESSED_DIR / "signbridge_model.pkl"
)

LABEL_ENCODER_FILE = (
    PROCESSED_DIR / "label_encoder.pkl"
)

X_TEST_FILE = (
    PROCESSED_DIR / "X_test.npy"
)

Y_TEST_FILE = (
    PROCESSED_DIR / "y_test.npy"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SIGNBRIDGE AI - IMPROVED MODEL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading trained model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Load encoder
    # --------------------------------------------------------

    label_encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    X_test = np.load(
        X_TEST_FILE
    )

    y_test = np.load(
        Y_TEST_FILE
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("GENERATING PREDICTIONS")
    print("-" * 60)

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    confidences = np.max(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # First 30 predictions
    # --------------------------------------------------------

    print("\nFirst 30 predictions:\n")

    print(
        "Actual".ljust(18)
        + "Predicted".ljust(20)
        + "Confidence".ljust(15)
        + "Result"
    )

    print("-" * 70)

    limit = min(
        30,
        len(X_test)
    )

    for i in range(limit):

        actual = label_encoder.inverse_transform(
            [y_test[i]]
        )[0]

        predicted = label_encoder.inverse_transform(
            [predictions[i]]
        )[0]

        confidence = (
            confidences[i] * 100
        )

        result = (
            "CORRECT"
            if y_test[i] == predictions[i]
            else "WRONG"
        )

        print(
            f"{actual:<18}"
            f"{predicted:<20}"
            f"{confidence:>7.2f}%      "
            f"{result}"
        )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    correct = int(
        np.sum(
            predictions == y_test
        )
    )

    total = len(
        y_test
    )

    print(
        f"\nCorrect predictions: "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"\nAverage confidence: "
        f"{np.mean(confidences) * 100:.2f}%"
    )

    print(
        f"Minimum confidence: "
        f"{np.min(confidences) * 100:.2f}%"
    )

    print(
        f"Maximum confidence: "
        f"{np.max(confidences) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confidence thresholds
    # --------------------------------------------------------

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:

        count = int(
            np.sum(
                confidences >= threshold
            )
        )

        print(
            f"Predictions >= "
            f"{threshold * 100:.0f}% confidence: "
            f"{count}/{total}"
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(
                len(label_encoder.classes_)
            ),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print(
        "CONFUSION MATRIX"
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(
            len(label_encoder.classes_)
        ),
    )

    print()

    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    print(
        matrix
    )

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("PER-CLASS ACCURACY")
    print("-" * 60)

    for index, label in enumerate(
        label_encoder.classes_
    ):

        mask = (
            y_test == index
        )

        total_class = int(
            np.sum(mask)
        )

        if total_class == 0:

            print(
                f"{label:<15} "
                f"No test samples"
            )

            continue

        correct_class = int(
            np.sum(
                predictions[mask] == index
            )
        )

        class_accuracy = (
            correct_class
            / total_class
            * 100
        )

        print(
            f"{label:<15} "
            f"{correct_class}/{total_class} "
            f"({class_accuracy:.2f}%)"
        )

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()