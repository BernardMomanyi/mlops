
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "data" / "football" / "processed" / "epl_features.csv"
)
OUTPUT_DIR = ROOT / "models" / "football"
OUTPUT_JSON = OUTPUT_DIR / "walk_forward_results.json"
OUTPUT_CSV = OUTPUT_DIR / "walk_forward_results.csv"

CLASSES = ["A", "D", "H"]

# Start with at least five seasons of training data.
MIN_TRAIN_SEASONS = 5

# These are the features used by the existing football model.
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


def build_preprocessor():
    """Create preprocessing that is fitted only on each training fold."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models():
    """Return fresh, unfitted model pipelines."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        ),
    }

    pipelines = {}

    for name, classifier in models.items():
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

    return pipelines


def evaluate_predictions(y_true, predictions, probabilities):
    """Calculate classification and probability metrics."""

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=CLASSES,
            )
        ),
    }


def evaluate_season(train_df, test_df, season):
    """Train on past seasons and evaluate the next season."""

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    if set(y_train.unique()) != set(CLASSES):
        raise ValueError(
            f"Training data before {season} does not contain all "
            f"expected classes {CLASSES}."
        )

    # Historical frequency baseline.
    frequencies = (
        y_train.value_counts(normalize=True)
        .reindex(CLASSES, fill_value=0.0)
    )

    frequency_probabilities = [
        float(frequencies[label]) for label in CLASSES
    ]

    frequency_matrix = [
        frequency_probabilities.copy()
        for _ in range(len(test_df))
    ]

    majority_class = y_train.value_counts().idxmax()
    majority_predictions = [majority_class] * len(test_df)

    frequency_predictions = [
        CLASSES[frequency_probabilities.index(
            max(frequency_probabilities)
        )]
    ] * len(test_df)

    baseline_results = {
        "Most-common-outcome baseline": {
            "accuracy": float(
                accuracy_score(y_test, majority_predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_test, majority_predictions
                )
            ),
            "log_loss": None,
        },
        "Historical-frequency baseline": {
            "accuracy": float(
                accuracy_score(y_test, frequency_predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_test, frequency_predictions
                )
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

    season_results = {
        "season": season,
        "train_matches": int(len(train_df)),
        "test_matches": int(len(test_df)),
        "train_start": train_df["Date"].min().strftime("%Y-%m-%d"),
        "train_end": train_df["Date"].max().strftime("%Y-%m-%d"),
        "test_start": test_df["Date"].min().strftime("%Y-%m-%d"),
        "test_end": test_df["Date"].max().strftime("%Y-%m-%d"),
        "baselines": baseline_results,
        "models": {},
    }

    # Fit each model from scratch using only past seasons.
    for model_name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        raw_probabilities = pipeline.predict_proba(X_test)

        fitted_classes = list(pipeline.classes_)
        column_indices = [
            fitted_classes.index(label) for label in CLASSES
        ]

        probabilities = raw_probabilities[:, column_indices]

        season_results["models"][model_name] = (
            evaluate_predictions(
                y_test,
                predictions,
                probabilities,
            )
        )

    return season_results


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    required_columns = set(FEATURES + ["Date", "season", "target"])
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    if df["Date"].isna().any():
        raise ValueError(
            "Dataset contains invalid dates. Fix them before evaluation."
        )

    if df["season"].isna().any() or df["target"].isna().any():
        raise ValueError(
            "Dataset contains missing season or target values."
        )

    unknown_labels = set(df["target"].unique()) - set(CLASSES)

    if unknown_labels:
        raise ValueError(
            f"Unexpected target labels: {sorted(unknown_labels)}"
        )

    # Explicit chronological season order.
    season_order = sorted(df["season"].unique())

    if len(season_order) <= MIN_TRAIN_SEASONS:
        raise ValueError(
            "Not enough seasons for walk-forward evaluation."
        )

    # Features must be available before the match.
    df = df.sort_values(["Date"]).reset_index(drop=True)

    # Require a minimum of five complete seasons for the first fold.
    test_seasons = season_order[MIN_TRAIN_SEASONS:]

    all_results = []

    print("=" * 76)
    print("EPL WALK-FORWARD EVALUATION")
    print("=" * 76)
    print(f"Dataset matches: {len(df)}")
    print(f"Available seasons: {len(season_order)}")
    print(f"Initial training seasons: {MIN_TRAIN_SEASONS}")
    print(f"Evaluation seasons: {len(test_seasons)}")

    for test_season in test_seasons:
        train_df = df[df["season"] < test_season].copy()
        test_df = df[df["season"] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        result = evaluate_season(
            train_df=train_df,
            test_df=test_df,
            season=test_season,
        )

        all_results.append(result)

        print("\n" + "-" * 76)
        print(f"Test season: {test_season}")
        print(
            f"Training matches: {len(train_df)} | "
            f"Test matches: {len(test_df)}"
        )

        for name, metrics in result["models"].items():
            print(f"\n{name}:")
            print(f"  Accuracy:          {metrics['accuracy']:.4f}")
            print(
                f"  Balanced accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )
            print(f"  Log loss:          {metrics['log_loss']:.4f}")

        for name, metrics in result["baselines"].items():
            print(f"\n{name}:")
            print(f"  Accuracy:          {metrics['accuracy']:.4f}")
            print(
                f"  Balanced accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )

            if metrics["log_loss"] is not None:
                print(f"  Log loss:          {metrics['log_loss']:.4f}")

    if not all_results:
        raise RuntimeError(
            "No seasons were evaluated. Check the season column."
        )

    # Summarize metrics across test seasons.
    summary = {}

    model_names = list(all_results[0]["models"].keys())
    baseline_names = list(all_results[0]["baselines"].keys())

    for name in model_names:
        summary[name] = {}

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "log_loss",
        ]:
            values = [
                result["models"][name][metric]
                for result in all_results
            ]

            summary[name][metric] = float(
                sum(values) / len(values)
            )

    for name in baseline_names:
        summary[name] = {}

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "log_loss",
        ]:
            values = [
                result["baselines"][name][metric]
                for result in all_results
                if result["baselines"][name][metric] is not None
            ]

            summary[name][metric] = (
                float(sum(values) / len(values))
                if values
                else None
            )

    output = {
        "evaluation": "expanding-window walk-forward",
        "initial_training_seasons": MIN_TRAIN_SEASONS,
        "feature_columns": FEATURES,
        "class_labels": CLASSES,
        "evaluated_seasons": [
            result["season"] for result in all_results
        ],
        "season_results": all_results,
        "mean_metrics_across_seasons": summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    # Produce a convenient flat CSV for analysis.
    csv_rows = []

    for result in all_results:
        for name, metrics in result["models"].items():
            csv_rows.append({
                "season": result["season"],
                "type": "model",
                "name": name,
                **metrics,
                "train_matches": result["train_matches"],
                "test_matches": result["test_matches"],
            })

        for name, metrics in result["baselines"].items():
            csv_rows.append({
                "season": result["season"],
                "type": "baseline",
                "name": name,
                **metrics,
                "train_matches": result["train_matches"],
                "test_matches": result["test_matches"],
            })

    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 76)
    print("MEAN METRICS ACROSS EVALUATED SEASONS")
    print("=" * 76)

    for name, metrics in summary.items():
        print(f"\n{name}:")
        print(f"  Mean accuracy:          {metrics['accuracy']:.4f}")
        print(
            f"  Mean balanced accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        if metrics["log_loss"] is not None:
            print(f"  Mean log loss:          {metrics['log_loss']:.4f}")

    print("\nResults saved:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print("\nWalk-forward evaluation completed successfully.")


if __name__ == "__main__":
    main()
