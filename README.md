# Champion--Challenger Model Comparison

## Champion Selection Rule

-   **Primary metric:** Accuracy.
-   **Minimum improvement margin:** The Challenger must achieve an
    accuracy at least **1 percentage point (0.01)** higher than the
    Champion to replace it.
-   **Tie-breaker:** If the Challenger's accuracy improvement is less
    than 1 percentage point, the model with the lower average inference
    latency remains the Champion.
-   **Metric direction:** Higher accuracy is better. Lower average
    inference latency is better.

Model A is the initial Champion, and Model B is the Challenger. This
rule is fixed before the final evaluation run and will not be changed
after inspecting the results.

## Dataset

The evaluation file is `data/eval_inputs.csv`, containing 100 labelled
examples. The separate prediction file is `data/new_data.csv`; it
contains 10 new, unlabelled examples and is reserved for the MLflow
prediction step.

## Dataset declaration

The evaluation examples are a custom-created sentiment dataset, not
examples downloaded from the official SST-2 dataset.
