
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = (
    PROJECT_ROOT / "data" / "football" / "processed"
    / "epl_features.csv"
)
MODEL_DIR = PROJECT_ROOT / "models" / "football"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "epl_result_model.joblib"
REPORT_FILE = MODEL_DIR / "epl_evaluation.json"

TARGET = "target"
CLASSES = ["A", "D", "H"]

NUMERIC_FEATURES = [
    "home_points_avg_5",
    "home_goals_for_avg_5",
    "home_goals_against_avg_5",
    "home_matches_prior",
    "home_win_rate_5",
    "home_draw_rate_5",
    "home_loss_rate_5",
    "away_points_avg_5",
    "away_goals_for_avg_5",
    "away_goals_against_avg_5",
    "away_matches_prior",
    "away_win_rate_5",
    "away_draw_rate_5",
    "away_loss_rate_5",
]

CATEGORICAL_FEATURES = ["HomeTeam", "AwayTeam"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run football/prepare_data.py first."
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])

    required = ["Date", "season", TARGET] + FEATURES
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Select only approved pre-match features and metadata.
    # Current-match goals are deliberately excluded.
    df = df[required].copy()
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"])
    df = df.reset_index(drop=True)

    if df[required].isna().any().any():
        raise ValueError("Required dataset fields contain missing values.")

    if not df[TARGET].isin(CLASSES).all():
        raise ValueError("Target must contain only H, D and A.")

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("Date column could not be parsed.")

    if not set(df["season"].unique()).issuperset({
        "2022-23", "2023-24", "2024-25"
    }):
        raise ValueError("Required development, validation or test seasons missing.")

    return df


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "teams",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def create_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate(pipeline, frame):
    X = frame[FEATURES]
    y = frame[TARGET]

    predictions = pipeline.predict(X)
    raw_probabilities = pipeline.predict_proba(X)
    model_classes = list(pipeline.named_steps["classifier"].classes_)

    probability_frame = pd.DataFrame(
        raw_probabilities,
        columns=model_classes,
    )
    probabilities = probability_frame.reindex(
        columns=CLASSES, fill_value=0.0
    ).to_numpy()

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "log_loss": float(log_loss(y, probabilities, labels=CLASSES)),
        "predictions": predictions,
        "probabilities": probabilities,
        "actual": y,
    }


def print_metrics(label, metrics):
    print(f"\n{label}")
    print("-" * 50)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Log loss:          {metrics['log_loss']:.4f}")


def main():
    df = load_data()

    development = df[df["season"] <= "2022-23"].copy()
    validation = df[df["season"] == "2023-24"].copy()
    final_test = df[df["season"] == "2024-25"].copy()

    if development.empty or validation.empty or final_test.empty:
        raise ValueError("One or more chronological data splits are empty.")

    for name, frame in [
        ("development", development),
        ("validation", validation),
        ("final test", final_test),
    ]:
        if frame[TARGET].nunique() < 3:
            raise ValueError(f"{name} split does not contain all result classes.")

    print("=" * 60)
    print("EPL MULTI-SEASON CHRONOLOGICAL EVALUATION")
    print("=" * 60)
    print(f"Total matches:      {len(df)}")
    print(f"Development:        {len(development)}")
    print(f"Validation:         {len(validation)}")
    print(f"Final test:         {len(final_test)}")
    print(
        f"Development dates:  {development['Date'].min().date()} "
        f"to {development['Date'].max().date()}"
    )
    print(
        f"Validation dates:   {validation['Date'].min().date()} "
        f"to {validation['Date'].max().date()}"
    )
    print(
        f"Final test dates:   {final_test['Date'].min().date()} "
        f"to {final_test['Date'].max().date()}"
    )

    # Fit candidates on development data and choose using validation only.
    candidates = {}

    for name, estimator in create_models().items():
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("classifier", estimator),
        ])
        pipeline.fit(development[FEATURES], development[TARGET])

        metrics = evaluate(pipeline, validation)
        candidates[name] = {"pipeline": pipeline, "validation": metrics}
        print_metrics(f"{name} — validation", metrics)

    best_name = min(
        candidates,
        key=lambda name: candidates[name]["validation"]["log_loss"],
    )
    print(f"\nSelected using validation log loss: {best_name}")

    # Refit selected model on development + validation.
    # The final test season has not been used for model selection.
    training_data = pd.concat([development, validation]).sort_values("Date")
    final_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", create_models()[best_name]),
    ])
    final_pipeline.fit(training_data[FEATURES], training_data[TARGET])

    test_metrics = evaluate(final_pipeline, final_test)
    print_metrics(f"{best_name} — untouched 2024-25 test", test_metrics)

    baseline_predictions = ["H"] * len(final_test)
    baseline_accuracy = float(
        accuracy_score(final_test[TARGET], baseline_predictions)
    )

    baseline_probabilities = [
        [0.0, 0.0, 1.0] for _ in range(len(final_test))
    ]
    baseline_loss = float(
        log_loss(final_test[TARGET], baseline_probabilities, labels=CLASSES)
    )

    print("\nAlways-home-win baseline on final test")
    print("-" * 50)
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Log loss: {baseline_loss:.4f}")

    print("\nFinal test classification report:")
    print(classification_report(
        test_metrics["actual"],
        test_metrics["predictions"],
        labels=CLASSES,
        target_names=["Away win", "Draw", "Home win"],
        zero_division=0,
    ))

    artifact = {
        "pipeline": final_pipeline,
        "features": FEATURES,
        "classes": CLASSES,
        "model_name": best_name,
        "validation_season": "2023-24",
        "final_test_season": "2024-25",
        "development_seasons": sorted(development["season"].unique()),
        "train_matches": len(training_data),
        "test_matches": len(final_test),
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "log_loss": test_metrics["log_loss"],
            "baseline_accuracy": baseline_accuracy,
            "baseline_log_loss": baseline_loss,
        },
    }

    joblib.dump(artifact, MODEL_FILE)

    report = {
        "model_name": best_name,
        "development_matches": len(development),
        "validation_matches": len(validation),
        "final_test_matches": len(final_test),
        "validation_metrics": {
            name: {
                key: value
                for key, value in item["validation"].items()
                if key in ("accuracy", "balanced_accuracy", "log_loss")
            }
            for name, item in candidates.items()
        },
        "final_test_metrics": artifact["test_metrics"],
        "features": FEATURES,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Selected model: {best_name}")
    print(f"Model saved:    {MODEL_FILE}")
    print(f"Report saved:   {REPORT_FILE}")


if __name__ == "__main__":
    main()
