import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

# -------------------- CONFIGURATION --------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"

MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B_ID = "textattack/bert-base-uncased-SST-2"

DATA_PATH = Path("data/eval_inputs.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK = "sentiment"
SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 8
DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "cuda" if DEVICE == 0 else "cpu"

STUDENT_USN = os.getenv("STUDENT_USN", "1MS25CI123")
DATASET_NAME = "Custom Sentiment Evaluation Dataset"

# Champion rule written in README before running this script:
# Challenger must improve accuracy by at least 0.01 (one percentage point).
# Otherwise, the faster model remains Champion.
PRIMARY_METRIC = "accuracy"
IMPROVEMENT_MARGIN = 0.01

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# -------------------- MLflow PYFUNC WRAPPER --------------------
class HuggingFaceSentimentModel(mlflow.pyfunc.PythonModel):
    """Loads a saved Hugging Face model directory and returns predictions."""

    def load_context(self, context):
        from transformers import pipeline as hf_pipeline

        self.classifier = hf_pipeline(
            "sentiment-analysis",
            model=context.artifacts["model_dir"],
            tokenizer=context.artifacts["model_dir"],
            device=-1,
        )

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(item) for item in model_input]

        results = self.classifier(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
        )

        rows = []
        for text, result in zip(texts, results):
            label = str(result["label"]).upper()
            if label in ("POSITIVE", "LABEL_1", "1"):
                prediction = 1
            elif label in ("NEGATIVE", "LABEL_0", "0"):
                prediction = 0
            else:
                raise ValueError(
                    f"Unrecognised sentiment label: {result['label']}. "
                    "Check the model's label mapping."
                )

            rows.append(
                {
                    "text": text,
                    "prediction": prediction,
                    "predicted_label": "positive" if prediction == 1 else "negative",
                    "score": float(result["score"]),
                }
            )

        return pd.DataFrame(rows)


def normalise_prediction(label):
    """Convert the Hugging Face pipeline label to 0=negative, 1=positive."""
    label = str(label).upper()

    if label in ("POSITIVE", "LABEL_1", "1"):
        return 1
    if label in ("NEGATIVE", "LABEL_0", "0"):
        return 0

    raise ValueError(
        f"Unrecognised label {label!r}. "
        "Inspect the model config label2id/id2label mapping before continuing."
    )


def evaluate_one_model(model_id, run_name, role, evaluation_df):
    print(f"\nLoading {model_id} ...")
    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        tokenizer=model_id,
        device=DEVICE,
    )

    texts = evaluation_df["text"].astype(str).tolist()

    start_time = time.perf_counter()
    raw_predictions = classifier(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )
    total_inference_s = time.perf_counter() - start_time

    predictions = [normalise_prediction(item["label"]) for item in raw_predictions]
    references = evaluation_df["label"].astype(int).tolist()

    metrics = {
        "accuracy": float(accuracy_score(references, predictions)),
        "precision": float(precision_score(references, predictions, zero_division=0)),
        "recall": float(recall_score(references, predictions, zero_division=0)),
        "f1": float(f1_score(references, predictions, zero_division=0)),
        "avg_latency_ms": float((total_inference_s * 1000) / len(texts)),
        "total_inference_s": float(total_inference_s),
    }

    prediction_df = pd.DataFrame(
        {
            "text": texts,
            "reference": references,
            "prediction": predictions,
            "reference_label": [
                "positive" if value == 1 else "negative" for value in references
            ],
            "predicted_label": [
                "positive" if value == 1 else "negative" for value in predictions
            ],
            "correct": [
                reference == prediction
                for reference, prediction in zip(references, predictions)
            ],
            "confidence": [float(item["score"]) for item in raw_predictions],
        }
    )

    short_name = run_name.replace("/", "-")
    prediction_path = OUTPUT_DIR / f"predictions_{short_name}.csv"
    metrics_path = OUTPUT_DIR / f"metrics_summary_{short_name}.json"
    prediction_df.to_csv(prediction_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Save the Hugging Face model locally so MLflow can package it as a pyfunc.
    safe_model_name = run_name.replace("/", "_").replace(" ", "_")
    model_dir = OUTPUT_DIR / f"hf_model_{safe_model_name}"
    classifier.model.save_pretrained(model_dir)
    classifier.tokenizer.save_pretrained(model_dir)

    transformers_version = __import__("transformers").__version__
    torch_version = torch.__version__
    mlflow_version = mlflow.__version__

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(
            {
                "hf_model_id": model_id,
                "task": TASK,
                "role": role,
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "eval_samples": len(evaluation_df),
                "seed": SEED,
                "device": DEVICE_NAME,
                "primary_metric": PRIMARY_METRIC,
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "student_usn": STUDENT_USN,
                "library_versions": (
                    f"transformers={transformers_version}; "
                    f"torch={torch_version}; mlflow={mlflow_version}"
                ),
                "dataset_name": DATASET_NAME,
                "hf_model_id": model_id,
                "role": role,
            }
        )

        mlflow.log_artifact(str(prediction_path))
        mlflow.log_artifact(str(metrics_path))

        # MLflow model uses the saved local model directory at prediction time.
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HuggingFaceSentimentModel(),
            artifacts={"model_dir": str(model_dir.resolve())},
            pip_requirements=[
                f"mlflow=={mlflow_version}",
                f"transformers=={transformers_version}",
                f"torch=={torch_version}",
                f"pandas=={pd.__version__}",
            ],
        )

        run_id = active_run.info.run_id

    # Per-model confusion matrix plot.
    cm = confusion_matrix(references, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix: {run_name}")
    fig.tight_layout()

    plot_path = OUTPUT_DIR / f"confusion_matrix_{short_name}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # Add the confusion matrix to the same run as an artifact.
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(plot_path))

    print(f"Finished {run_name}: accuracy={metrics['accuracy']:.4f}")
    return {
        "run_id": run_id,
        "model_id": model_id,
        "run_name": run_name,
        "predictions": predictions,
        "prediction_df": prediction_df,
        "metrics": metrics,
    }


# -------------------- MAIN EVALUATION --------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Put eval_inputs.csv inside the data folder."
        )

    evaluation_df = pd.read_csv(DATA_PATH)

    required_columns = {"text", "label"}
    if not required_columns.issubset(evaluation_df.columns):
        raise ValueError("eval_inputs.csv must contain columns named 'text' and 'label'.")

    evaluation_df = evaluation_df.dropna(subset=["text", "label"]).copy()
    evaluation_df["text"] = evaluation_df["text"].astype(str)
    evaluation_df["label"] = evaluation_df["label"].astype(int)

    if not set(evaluation_df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 (negative) or 1 (positive).")

    if len(evaluation_df) == 0:
        raise ValueError("The evaluation dataset is empty.")

    # Save the exact shared evaluation inputs in outputs as well.
    saved_eval_path = OUTPUT_DIR / "eval_inputs.csv"
    evaluation_df.to_csv(saved_eval_path, index=False)

    result_a = evaluate_one_model(
        MODEL_A_ID,
        "A_sentiment_distilbert-sst2",
        "champion_candidate",
        evaluation_df,
    )
    result_b = evaluate_one_model(
        MODEL_B_ID,
        "B_sentiment_textattack-bert-sst2",
        "challenger_candidate",
        evaluation_df,
    )

    # Compare models using the fixed rule.
    accuracy_delta = (
        result_b["metrics"]["accuracy"] - result_a["metrics"]["accuracy"]
    )

    if accuracy_delta >= IMPROVEMENT_MARGIN:
        winner = "B_sentiment_textattack-bert-sst2"
        decision = "Challenger replaces Champion"
    else:
        # If accuracy improvement is below the margin, select the faster model.
        latency_a = result_a["metrics"]["avg_latency_ms"]
        latency_b = result_b["metrics"]["avg_latency_ms"]
        winner = (
            result_a["run_name"]
            if latency_a <= latency_b
            else result_b["run_name"]
        )
        decision = "Accuracy margin not reached; faster model selected"

    summary = {
        "model_a": {
            "run_id": result_a["run_id"],
            "hf_model_id": result_a["model_id"],
            **result_a["metrics"],
        },
        "model_b": {
            "run_id": result_b["run_id"],
            "hf_model_id": result_b["model_id"],
            **result_b["metrics"],
        },
        "primary_metric": PRIMARY_METRIC,
        "margin": IMPROVEMENT_MARGIN,
        "delta_primary_metric": float(accuracy_delta),
        "winner": winner,
        "decision": decision,
    }

    summary_path = OUTPUT_DIR / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # Bar chart comparing both models on every metric.
    metric_names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        [result_a["metrics"][metric] for metric in metric_names],
        width,
        label="Model A",
    )
    ax.bar(
        x + width / 2,
        [result_b["metrics"][metric] for metric in metric_names],
        width,
        label="Model B",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Champion vs Challenger Metrics")
    ax.legend()
    fig.tight_layout()

    comparison_plot = OUTPUT_DIR / "model_comparison.png"
    fig.savefig(comparison_plot, dpi=150)
    plt.close(fig)

    # Save five examples where the models disagree.
    disagreement = result_a["prediction_df"][
        result_a["prediction_df"]["prediction"] != result_b["prediction_df"]["prediction"]
    ].copy()
    disagreement["model_b_prediction"] = result_b["prediction_df"]["prediction"]
    disagreement["model_b_label"] = result_b["prediction_df"]["predicted_label"]

    disagreement_path = OUTPUT_DIR / "model_disagreements.csv"
    disagreement.head(5).to_csv(disagreement_path, index=False)

    # Summary/comparison MLflow run.
    with mlflow.start_run(run_name="compare_sentiment"):
        mlflow.log_params(
            {
                "primary_metric": PRIMARY_METRIC,
                "margin": IMPROVEMENT_MARGIN,
                "model_a_run_id": result_a["run_id"],
                "model_b_run_id": result_b["run_id"],
                "eval_samples": len(evaluation_df),
                "seed": SEED,
            }
        )
        mlflow.log_metric("delta_primary_metric", float(accuracy_delta))
        mlflow.set_tag("winner", winner)
        mlflow.set_tag("decision", decision)
        mlflow.log_artifact(str(saved_eval_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(comparison_plot))
        mlflow.log_artifact(str(disagreement_path))
        mlflow.log_artifact("README.md")
        mlflow.log_artifact("requirements.txt")

    print("\n========== COMPARISON ==========")
    print(pd.DataFrame(
        {
            "Model A": result_a["metrics"],
            "Model B": result_b["metrics"],
        }
    ).round(4))
    print(f"\nAccuracy difference (B - A): {accuracy_delta:.4f}")
    print(f"Winner: {winner}")
    print(f"Decision: {decision}")
    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")
    print("Open http://127.0.0.1:5000 to view MLflow.")


if __name__ == "__main__":
    main()
