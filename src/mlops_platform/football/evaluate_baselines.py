
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "data" / "football" / "processed" / "epl_features.csv"
)
MODEL_PATH = ROOT / "models" / "football" / "epl_result_model.joblib"
REPORT_PATH = ROOT / "models" / "football" / "epl_evaluation.json"

# Football-Data result labels:
# A = Away win, D = Draw, H = Home win
CLASSES = ["A", "D", "H"]

DEVELOPMENT_END_SEASON = "2022-23"
VALIDATION_SEASON = "2023-24"
TEST_SEASON = "2024-25"


def validate_files():
    """Make sure the required project files exist."""
    required_files = [DATA_PATH, MODEL_PATH, REPORT_PATH]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )


def main():
    validate_files()

    # ---------------------------------------------------------
    # 1. Load the processed data and existing trained model
    # ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    artifact = joblib.load(MODEL_PATH)

    required_artifact_keys = {"pipeline", "features", "classes"}
    missing_keys = required_artifact_keys - set(artifact.keys())

    if missing_keys:
        raise KeyError(
            f"Model artifact is missing keys: {sorted(missing_keys)}"
        )

    required_columns = {"season", "target", "Date"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 2. Re-create the chronological season splits
    # ---------------------------------------------------------
    development = df[
        df["season"] <= DEVELOPMENT_END_SEASON
    ].copy()

    validation = df[
        df["season"] == VALIDATION_SEASON
    ].copy()

    test = df[
        df["season"] == TEST_SEASON
    ].copy()

    if development.empty or validation.empty or test.empty:
        raise ValueError(
            "One or more expected season splits are empty. "
            "Check the season values in epl_features.csv."
        )

    # Validate the target labels.
    for split_name, split_df in [
        ("development", development),
        ("validation", validation),
        ("test", test),
    ]:
        unknown_labels = set(split_df["target"].dropna()) - set(CLASSES)

        if unknown_labels:
            raise ValueError(
                f"Unexpected target labels in {split_name}: "
                f"{sorted(unknown_labels)}"
            )

        if split_df["target"].isna().any():
            raise ValueError(
                f"The {split_name} split contains missing target labels."
            )

    if test["Date"].isna().any():
        raise ValueError("The test split contains invalid dates.")

    # ---------------------------------------------------------
    # 3. Generate predictions using the existing model
    # ---------------------------------------------------------
    feature_columns = artifact["features"]
    model = artifact["pipeline"]
    model_classes = list(artifact["classes"])

    missing_features = set(feature_columns) - set(test.columns)

    if missing_features:
        raise ValueError(
            f"Test data is missing model features: "
            f"{sorted(missing_features)}"
        )

    if set(model_classes) != set(CLASSES):
        raise ValueError(
            f"Expected model classes {CLASSES}, "
            f"but artifact contains {model_classes}."
        )

    X_test = test[feature_columns]
    y_test = test["target"]

    model_predictions = model.predict(X_test)
    raw_probabilities = model.predict_proba(X_test)

    # Align probability columns with the explicitly defined label order.
    pipeline_classes = list(model.classes_)

    if set(pipeline_classes) != set(CLASSES):
        raise ValueError(
            f"Unexpected pipeline classes: {pipeline_classes}"
        )

    pipeline_column_indices = [
        pipeline_classes.index(label) for label in CLASSES
    ]

    model_probabilities = raw_probabilities[
        :, pipeline_column_indices
    ]

    # ---------------------------------------------------------
    # 4. Baseline 1: always predict the most common outcome
    # ---------------------------------------------------------
    majority_class = development["target"].mode().iloc[0]

    majority_predictions = [
        majority_class
    ] * len(test)

    # ---------------------------------------------------------
    # 5. Baseline 2: use historical outcome frequencies
    # ---------------------------------------------------------
    class_frequencies = (
        development["target"]
        .value_counts(normalize=True)
        .reindex(CLASSES, fill_value=0.0)
    )

    frequency_probabilities = [
        float(class_frequencies[label])
        for label in CLASSES
    ]

    # Repeat the historical probability distribution for every match.
    frequency_matrix = [
        frequency_probabilities.copy()
        for _ in range(len(test))
    ]

    # The most likely historical outcome is the prediction for every match.
    most_frequent_index = frequency_probabilities.index(
        max(frequency_probabilities)
    )

    frequency_predictions = [
        CLASSES[most_frequent_index]
    ] * len(test)

    # ---------------------------------------------------------
    # 6. Calculate evaluation metrics
    # ---------------------------------------------------------
    results = {
        "test_season": TEST_SEASON,
        "development_end_season": DEVELOPMENT_END_SEASON,
        "validation_season": VALIDATION_SEASON,
        "development_matches": int(len(development)),
        "validation_matches": int(len(validation)),
        "test_matches": int(len(test)),
        "test_start_date": test["Date"].min().strftime("%Y-%m-%d"),
        "test_end_date": test["Date"].max().strftime("%Y-%m-%d"),
        "majority_class": str(majority_class),
        "development_class_frequencies": {
            label: float(class_frequencies[label])
            for label in CLASSES
        },
        "model": {
            "accuracy": float(
                accuracy_score(y_test, model_predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, model_predictions)
            ),
            "log_loss": float(
                log_loss(
                    y_test,
                    model_probabilities,
                    labels=CLASSES,
                )
            ),
        },
        "majority_baseline": {
            "accuracy": float(
                accuracy_score(y_test, majority_predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, majority_predictions)
            ),
        },
        "historical_frequency_baseline": {
            "accuracy": float(
                accuracy_score(y_test, frequency_predictions)
            ),
            "log_loss": float(
                log_loss(
                    y_test,
                    frequency_matrix,
                    labels=CLASSES,
                )
            ),
        },
    }

    # ---------------------------------------------------------
    # 7. Update the evaluation report without deleting old data
    # ---------------------------------------------------------
    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    report["baseline_comparison"] = results

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # ---------------------------------------------------------
    # 8. Display the results
    # ---------------------------------------------------------
    print("=" * 64)
    print("EPL BASELINE COMPARISON")
    print("=" * 64)

    print(f"Development matches: {len(development)}")
    print(f"Validation matches:  {len(validation)}")
    print(f"Test matches:        {len(test)}")
    print(
        f"Test period:         "
        f"{results['test_start_date']} to {results['test_end_date']}"
    )
    print(f"Majority class:      {majority_class}")

    print("\nDevelopment outcome frequencies:")

    for label in CLASSES:
        print(
            f"  {label}: "
            f"{class_frequencies[label]:.4f}"
        )

    print("\nTrained model:")
    print(f"  Accuracy:          {results['model']['accuracy']:.4f}")
    print(
        f"  Balanced accuracy: "
        f"{results['model']['balanced_accuracy']:.4f}"
    )
    print(f"  Log loss:          {results['model']['log_loss']:.4f}")

    print("\nMost-common-outcome baseline:")
    print(
        f"  Accuracy:          "
        f"{results['majority_baseline']['accuracy']:.4f}"
    )
    print(
        f"  Balanced accuracy: "
        f"{results['majority_baseline']['balanced_accuracy']:.4f}"
    )

    print("\nHistorical-frequency baseline:")
    print(
        f"  Accuracy:          "
        f"{results['historical_frequency_baseline']['accuracy']:.4f}"
    )
    print(
        f"  Log loss:          "
        f"{results['historical_frequency_baseline']['log_loss']:.4f}"
    )

    print("\nUpdated evaluation report:")
    print(REPORT_PATH)
    print("\nBaseline evaluation completed successfully.")


if __name__ == "__main__":
    main()
