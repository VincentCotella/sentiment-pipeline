# -*- coding: utf-8 -*-
"""Train a sentiment analysis model using scikit-learn and MLflow."""

import argparse
import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_DATA_PATH = "data/clean/tweets.parquet"
DEFAULT_MODEL_PATH = "models/model.joblib"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument(
        "--experiment",
        default="sentiment",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=2000,
        help="Maximum number of TF-IDF features",
    )
    parser.add_argument(
        "--ngram_range",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="1,2",
        help="N-gram range as 'min,max'",
    )
    parser.add_argument(
        "--tracking_uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI",
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to dataset")
    parser.add_argument(
        "--model_out", default=DEFAULT_MODEL_PATH, help="Output path for the model"
    )
    return parser.parse_args()


def load_dataset(path: str) -> tuple[pd.Series, pd.Series]:
    """Load dataset and return features and labels."""
    df = pd.read_parquet(path)
    if "label" in df.columns:
        y = df["label"]
    elif "sentiment" in df.columns:
        y = df["sentiment"]
    else:
        raise KeyError("Dataset must contain 'label' or 'sentiment' column")

    if "clean_text" not in df.columns:
        raise KeyError("Dataset must contain 'clean_text' column")
    X = df["clean_text"]
    return X, y


def build_grid(max_features: int, ngram_range: tuple[int, int]) -> GridSearchCV:
    """Create a GridSearchCV object for the pipeline."""
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=ngram_range),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, multi_class="multinomial"),
            ),
        ]
    )

    param_grid = {"clf__C": [0.1, 1.0, 10.0]}
    return GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1_macro",
    )


def evaluate(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> tuple[float, float, plt.Figure]:
    """Evaluate the model and return metrics and confusion matrix figure."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(cm.shape[1]),
        yticks=range(cm.shape[0]),
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return acc, f1, fig


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    X, y = load_dataset(args.data)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    grid = build_grid(args.max_features, args.ngram_range)

    with mlflow.start_run():
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        mlflow.log_params(grid.best_params_)
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("ngram_range", args.ngram_range)

        acc, f1, fig = evaluate(best_model, X_val, y_val)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        model_path = Path(args.model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(str(model_path))

        print(f"Accuracy: {acc:.4f}")
        print(f"F1-macro: {f1:.4f}")


if __name__ == "__main__":
    main()
