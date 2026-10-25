
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/football/processed/epl_features.csv"
MODEL_PATH = ROOT / "models/football/epl_result_model.joblib"
RESULTS_PATH = ROOT / "models/football/walk_forward_results.json"

CLASSES = ["A", "D", "H"]


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Walk-forward results not found: {RESULTS_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    print("=" * 65)
    print("WALK-FORWARD SEASON STABILITY")
    print("=" * 65)

    for season in results["season_results"]:
        print(f"\n{season['season']}")

        for name, metrics in season["models"].items():
            print(
                f"  {name}: "
                f"accuracy={metrics['accuracy']:.3f}, "
                f"balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
                f"log_loss={metrics['log_loss']:.3f}"
            )

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    artifact = joblib.load(MODEL_PATH)

    test = df[df["season"] == artifact["final_test_season"]].copy()

    if test.empty:
        raise ValueError("The final test season was not found.")

    X_test = test[artifact["features"]]
    y_test = test["target"]

    predictions = artifact["pipeline"].predict(X_test)

    print("\n" + "=" * 65)
    print(f"FINAL-TEST CLASSIFICATION: {artifact['final_test_season']}")
    print("=" * 65)

    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASSES,
            target_names=["Away win", "Draw", "Home win"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=CLASSES),
            index=["Actual Away", "Actual Draw", "Actual Home"],
            columns=["Predicted Away", "Predicted Draw", "Predicted Home"],
        )
    )


if __name__ == "__main__":
    main()
