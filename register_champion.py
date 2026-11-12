"""
Register the Champion and Challenger models in the MLflow Model Registry.

Run this after evaluate.py has completed and both model runs are visible in MLflow.

The script:
1. Finds the latest Model A and Model B evaluation runs.
2. Applies the same decision rule used by evaluate.py:
   - Challenger replaces Champion if its accuracy is at least 0.01 higher.
   - Otherwise, the faster model is selected.
3. Registers both logged models under one registered-model name.
4. Assigns the aliases "champion" and "challenger".
"""

import os
import sys
import time

import mlflow
from mlflow.tracking import MlflowClient


# Keep these settings consistent with evaluate.py.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-classifier"
ACCURACY_MARGIN = 0.01

MODEL_A_RUN_NAME = "A_sentiment_distilbert-sst2"
MODEL_B_RUN_NAME = "B_sentiment_textattack-bert-sst2"


def get_latest_run(client, experiment_id, run_name):
    """Return the latest finished run with the requested MLflow run name."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )

    for run in runs:
        if run.info.status == "FINISHED" and "accuracy" in run.data.metrics:
            return run

    raise RuntimeError(
        f"No finished run with an accuracy metric was found for '{run_name}'. "
        "Run evaluate.py successfully before running this script."
    )


def choose_champion(run_a, run_b):
    """Apply the fixed accuracy-margin rule, using latency as the tie-breaker."""
    accuracy_a = float(run_a.data.metrics["accuracy"])
    accuracy_b = float(run_b.data.metrics["accuracy"])

    latency_a = float(run_a.data.metrics.get("avg_latency_ms", float("inf")))
    latency_b = float(run_b.data.metrics.get("avg_latency_ms", float("inf")))

    if accuracy_b >= accuracy_a + ACCURACY_MARGIN:
        return run_b, run_a, "Model B wins by the required accuracy margin."

    if accuracy_a >= accuracy_b + ACCURACY_MARGIN:
        return run_a, run_b, "Model A wins by the required accuracy margin."

    if latency_b < latency_a:
        return run_b, run_a, "Accuracy difference is below the margin; Model B is faster."

    return run_a, run_b, "Accuracy difference is below the margin; Model A is faster."


def register_run_model(client, run):
    """Register the pyfunc model logged at the run's 'model' artifact path."""
    model_uri = f"runs:/{run.info.run_id}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    # Wait briefly until the new model version is ready.
    for _ in range(30):
        version = client.get_model_version(
            name=REGISTERED_MODEL_NAME,
            version=registered.version,
        )
        if version.status == "READY":
            return str(registered.version)
        if version.status == "FAILED":
            raise RuntimeError(
                f"Registration failed for run {run.info.run_id}: "
                f"{version.status_message}"
            )
        time.sleep(2)

    raise TimeoutError(
        f"Model version {registered.version} did not become READY in time."
    )


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found. "
            "Check that the MLflow server is running and evaluate.py has completed."
        )

    run_a = get_latest_run(client, experiment.experiment_id, MODEL_A_RUN_NAME)
    run_b = get_latest_run(client, experiment.experiment_id, MODEL_B_RUN_NAME)

    champion_run, challenger_run, decision = choose_champion(run_a, run_b)

    print("\nEvaluation results")
    print("------------------")
    print(
        f"Model A accuracy: {run_a.data.metrics['accuracy']:.4f} | "
        f"latency: {run_a.data.metrics.get('avg_latency_ms', float('nan')):.2f} ms"
    )
    print(
        f"Model B accuracy: {run_b.data.metrics['accuracy']:.4f} | "
        f"latency: {run_b.data.metrics.get('avg_latency_ms', float('nan')):.2f} ms"
    )
    print(f"\nDecision: {decision}")
    print(f"Champion run: {champion_run.info.run_id}")
    print(f"Challenger run: {challenger_run.info.run_id}")

    # Register both models first, then assign the requested aliases.
    champion_version = register_run_model(client, champion_run)
    challenger_version = register_run_model(client, challenger_run)

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=champion_version,
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="challenger",
        version=challenger_version,
    )

    # Add useful metadata to the registered model versions.
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, champion_version, "role", "champion"
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, challenger_version, "role", "challenger"
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, champion_version, "source_run_id",
        champion_run.info.run_id
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, challenger_version, "source_run_id",
        challenger_run.info.run_id
    )

    print("\nModel Registry updated successfully")
    print("------------------------------------")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")
    print(f"Champion version: {champion_version} (alias: champion)")
    print(f"Challenger version: {challenger_version} (alias: challenger)")
    print("\nOpen the MLflow UI → Models → sentiment-classifier to verify the aliases.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
