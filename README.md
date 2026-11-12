# Champion--Challenger Model Comparison with Hugging Face and MLflow

## 1. Task and Models

**Task:** Binary sentiment classification (positive / negative).

**Evaluation dataset:** `data/eval_inputs.csv` --- 100 labelled
examples. Both models are evaluated on the same saved inputs and labels.

**New prediction dataset:** `data/new_data.csv` --- 10 unlabelled
examples used for inference after model registration.

  ---------------------------------------------------------------------------------------
  Role                                Hugging Face model ID
  ----------------------------------- ---------------------------------------------------
  Model A --- Initial Champion        `distilbert-base-uncased-finetuned-sst-2-english`

  Model B --- Challenger              `textattack/bert-base-uncased-SST-2`
  ---------------------------------------------------------------------------------------

The evaluation dataset is custom-created. It is not the official SST-2
validation or test split.

## 2. How to Run

Run all commands from the project root directory using Windows
PowerShell.

### 2.1 Create and activate a virtual environment

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2.2 Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Start the MLflow tracking server

Open a **second terminal**, activate the same virtual environment, and
run:

``` powershell
.\venv\Scripts\Activate.ps1
python -m mlflow server --host 127.0.0.1 --port 5000
```

Keep this terminal running. Open the MLflow UI at:

`http://127.0.0.1:5000`

### 2.4 Evaluate both models

Return to the project terminal and run:

``` powershell
python evaluate.py
```

This evaluates both models on `data/eval_inputs.csv`, logs parameters,
metrics and model artifacts to MLflow, and saves evaluation outputs in
`outputs/`.

### 2.5 Register the models and assign aliases

``` powershell
python register_champion.py
```

This registers both models under `sentiment-classifier` and assigns the
`champion` and `challenger` aliases according to the decision rule in
Section 3.

### 2.6 Predict using the Champion alias

``` powershell
python predict_from_mlflow.py
```

The script loads the model through the MLflow URI:

``` text
models:/sentiment-classifier@champion
```

It displays the Champion alias, registered model version, Hugging Face
model ID and 10 predictions. It saves the results to
`outputs/new_data_predictions.csv`.

## 3. Champion Selection Rule

The decision rule was defined before the final evaluation run and was
not changed after inspecting the results.

-   **Primary metric:** Accuracy --- higher is better.
-   **Minimum improvement margin:** The Challenger must improve accuracy
    by at least `0.01` (one percentage point) to replace the current
    Champion.
-   **Tie-breaker:** If the accuracy improvement is below `0.01`, the
    model with lower average inference latency is preferred.
-   **Latency direction:** Lower average inference latency is better.

## 4. Evaluation Results

Both models were evaluated on the same 100 labelled examples using CPU
inference.

  -----------------------------------------------------------------------
  Metric                Model A ---        Model B --- Difference (B − A)
                         DistilBERT    TextAttack BERT 
  -------------- ------------------ ------------------ ------------------
  Accuracy                   0.9700             0.9900            +0.0200

  Precision                  0.9796             1.0000            +0.0204

  Recall                     0.9600             0.9800            +0.0200

  F1-score                   0.9697             0.9899            +0.0202

  Average                   12.8654            26.8266           +13.9612
  latency (ms)                                         

  Total                      1.2865             2.6827            +1.3962
  inference time                                       
  (s)                                                  
  -----------------------------------------------------------------------

**Selected Champion:** Model B --- `textattack/bert-base-uncased-SST-2`.

Model B achieved 0.99 accuracy compared with Model A's 0.97. Its
improvement of 0.02 exceeds the required margin of 0.01.

## 5. Decision

Model B, `textattack/bert-base-uncased-SST-2`, is selected as the
Champion. It achieved an accuracy of 0.99, compared with 0.97 for Model
A, `distilbert-base-uncased-finetuned-sst-2-english`. The absolute
accuracy improvement is 0.02, or two percentage points, so Model B
exceeds the pre-defined minimum improvement margin of 0.01. Model B also
produced higher precision, recall and F1-score on this evaluation set.

There is a latency trade-off. Model B required approximately 26.83 ms
per example, while Model A required approximately 12.87 ms. Therefore,
Model A was faster in this CPU evaluation. Under the decision rule, the
accuracy gain takes priority when it reaches the required margin, so
Model B replaces Model A.

The result should be interpreted cautiously. The evaluation contains
only 100 examples and uses a custom-created dataset rather than the
official SST-2 evaluation split. A two-percentage-point difference on a
sample of this size may not generalise to other data. The models may
also behave differently on text from domains unlike their fine-tuning
data. A larger, representative, independently labelled test set and
repeated latency measurements would provide stronger evidence.

## 6. Limitations and Declaration

-   The evaluation set contains 100 labelled examples; results may vary
    on a larger or different dataset.
-   The dataset is custom-created and is not the official SST-2
    benchmark split.
-   Both selected models are SST-2 sentiment classifiers; performance
    may be weaker on text from domains unlike their training data.
-   Latency was measured on CPU in the local environment and may differ
    on other hardware or under different system load.
-   Individual predictions can be incorrect even when overall evaluation
    metrics are high.
-   **Tools used:** Python, Hugging Face Transformers, PyTorch, MLflow,
    pandas, scikit-learn, matplotlib, Git and GitHub.
-   **AI assistance declaration:** ChatGPT was used to help draft and
    debug code and documentation. The scripts were run, and the reported
    results and screenshots were checked in the local project
    environment.
