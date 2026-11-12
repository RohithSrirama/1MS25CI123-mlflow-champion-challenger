"""
Champion inference using MLflow Registry alias.
Reads data/new_data.csv by default and writes outputs/new_data_predictions.csv.
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = "sentiment-classifier"
CHAMPION_ALIAS = "champion"
NEW_DATA_PATH = Path("data") / "new_data.csv"
OUTPUT_PATH = Path("outputs") / "new_data_predictions.csv"


def load_texts(cli_texts):
    if cli_texts:
        texts = cli_texts
    else:
        if not NEW_DATA_PATH.exists():
            raise FileNotFoundError(f"Missing {NEW_DATA_PATH}")
        df = pd.read_csv(NEW_DATA_PATH)
        if "text" not in df.columns:
            raise ValueError("data/new_data.csv must have a 'text' column.")
        texts = df["text"].dropna().astype(str).tolist()

    texts = [str(x).strip() for x in texts if str(x).strip()]
    if not texts:
        raise ValueError("No non-empty texts to predict.")
    return texts


def normalize_label(value):
    value = str(value).strip()
    upper = value.upper()
    if upper in {"POSITIVE", "LABEL_1", "1"}:
        return "POSITIVE"
    if upper in {"NEGATIVE", "LABEL_0", "0"}:
        return "NEGATIVE"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="*", help="Optional texts to classify.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"

    print("Loading Champion model from MLflow...")
    print(f"Tracking URI : {TRACKING_URI}")
    print(f"Model URI    : {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        raise RuntimeError(
            "Could not load Champion alias. Check the MLflow server and run "
            "register_champion.py first."
        ) from exc

    texts = load_texts(args.texts)
    predictions = model.predict(pd.DataFrame({"text": texts}))

    if isinstance(predictions, pd.DataFrame):
        raw = (predictions["prediction"].tolist()
               if "prediction" in predictions.columns
               else predictions.iloc[:, 0].tolist())
    elif isinstance(predictions, pd.Series):
        raw = predictions.tolist()
    else:
        raw = list(predictions)

    if len(raw) != len(texts):
        raise RuntimeError("Number of predictions does not match input texts.")

    labels = [normalize_label(x) for x in raw]
    result = pd.DataFrame({
        "text": texts,
        "predicted_sentiment": labels,
        "model_alias": CHAMPION_ALIAS,
        "model_name": REGISTERED_MODEL_NAME,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\nChampion predictions")
    print("=" * 72)
    for i, (text, label) in enumerate(zip(texts, labels), start=1):
        print(f"\n{i}. Text: {text}")
        print(f"   Predicted sentiment: {label}")

    print("\n" + "=" * 72)
    print(f"Completed {len(texts)} predictions using the MLflow Champion alias.")
    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
