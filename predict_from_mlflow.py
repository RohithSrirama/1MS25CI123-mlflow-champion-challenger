"""
Load the Champion model through its MLflow Registry alias, display its
registered version and Hugging Face model ID, predict 10 new examples, and
save the predictions to outputs/new_data_predictions.csv.

Run from the project root:
    python predict_from_mlflow.py
"""

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = "sentiment-classifier"
CHAMPION_ALIAS = "champion"
NEW_DATA_PATH = Path("data") / "new_data.csv"
OUTPUT_PATH = Path("outputs") / "new_data_predictions.csv"


def load_input_texts():
    if not NEW_DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {NEW_DATA_PATH}")

    dataframe = pd.read_csv(NEW_DATA_PATH)
    if "text" not in dataframe.columns:
        raise ValueError(f"{NEW_DATA_PATH} must contain a 'text' column.")

    texts = dataframe["text"].dropna().astype(str).str.strip().tolist()
    texts = [text for text in texts if text]

    if len(texts) != 10:
        raise ValueError(
            f"Expected exactly 10 non-empty examples in {NEW_DATA_PATH}; "
            f"found {len(texts)}."
        )
    return texts


def normalize_prediction(prediction):
    value = str(prediction).strip()
    upper = value.upper()
    if upper in {"POSITIVE", "LABEL_1", "1"}:
        return "POSITIVE"
    if upper in {"NEGATIVE", "LABEL_0", "0"}:
        return "NEGATIVE"
    return value


def get_champion_metadata(client):
    version_info = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
    )

    hf_model_id = version_info.tags.get("hf_model_id")
    source_run_id = version_info.tags.get("source_run_id")

    # register_champion.py stores source_run_id; evaluate.py logs hf_model_id
    # as a run parameter, so retrieve it from the source run when needed.
    if not hf_model_id and source_run_id:
        source_run = client.get_run(source_run_id)
        hf_model_id = source_run.data.params.get("hf_model_id")

    if not hf_model_id:
        hf_model_id = (
            "Not recorded in MLflow model-version tags or source-run parameters"
        )

    return str(version_info.version), hf_model_id


def extract_predictions(predictions):
    if isinstance(predictions, pd.DataFrame):
        if "prediction" in predictions.columns:
            return predictions["prediction"].tolist()
        return predictions.iloc[:, 0].tolist()
    if isinstance(predictions, pd.Series):
        return predictions.tolist()
    return list(predictions)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"

    print("=" * 72)
    print("CHAMPION MODEL INFORMATION")
    print("=" * 72)
    print(f"Registered model : {REGISTERED_MODEL_NAME}")
    print(f"Champion alias   : @{CHAMPION_ALIAS}")
    print(f"MLflow model URI : {model_uri}")

    try:
        model_version, hf_model_id = get_champion_metadata(client)
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Champion model or retrieve its metadata. "
            "Check that the MLflow server is running and that the "
            "'champion' alias exists. Run register_champion.py first."
        ) from exc

    print(f"Registered version: {model_version}")
    print(f"Hugging Face ID   : {hf_model_id}")

    texts = load_input_texts()
    predictions = extract_predictions(
        model.predict(pd.DataFrame({"text": texts}))
    )

    if len(predictions) != len(texts):
        raise RuntimeError(
            f"Received {len(predictions)} predictions for {len(texts)} inputs."
        )

    labels = [normalize_prediction(value) for value in predictions]
    results = pd.DataFrame({
        "text": texts,
        "predicted_sentiment": labels,
        "model_alias": [CHAMPION_ALIAS] * len(texts),
        "registered_model": [REGISTERED_MODEL_NAME] * len(texts),
        "model_version": [model_version] * len(texts),
        "hf_model_id": [hf_model_id] * len(texts),
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 72)
    print("CHAMPION PREDICTIONS")
    print("=" * 72)
    for index, (text, label) in enumerate(zip(texts, labels), start=1):
        print(f"\n{index}. Text: {text}")
        print(f"   Predicted sentiment: {label}")

    print("\n" + "=" * 72)
    print(f"Completed {len(texts)} predictions using @{CHAMPION_ALIAS}.")
    print(f"Registered model version: {model_version}")
    print(f"Hugging Face model ID: {hf_model_id}")
    print(f"Saved CSV: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
