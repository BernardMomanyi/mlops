
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train() -> None:
    # 1. Load the built-in dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3. Create the machine learning pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    )

    # 4. Configure MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mlops-platform-classification")

    # 5. Train and track the model
    with mlflow.start_run():
        model.fit(X_train, y_train)

        # 6. Generate predictions
        predictions = model.predict(X_test)

        # 7. Evaluate model performance
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # 8. Record parameters and metrics in MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # 9. Save the trained model as an MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(3),
        )

        # 10. Display results
        print("\nTraining completed successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train()
