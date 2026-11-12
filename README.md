# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Task and Models

**Task:** Binary sentiment classification (positive / negative).

**Evaluation dataset:** `data/eval_inputs.csv`, containing 100 labelled examples. Both models are evaluated on the exact same saved inputs and labels.

**New prediction dataset:** `data/new_data.csv`, containing 10 unlabelled examples. These examples are used only after model registration to demonstrate inference through the MLflow Champion alias.

| Role | Hugging Face model ID |
|---|---|
| Model A — Initial Champion | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Model B — Challenger | `textattack/bert-base-uncased-SST-2` |

The evaluation dataset is a custom-created sentiment dataset. It is not the official SST-2 validation or test split.

## 2. How to Run

Run the commands from the project root directory.

### 2.1 Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2.2 Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Start the MLflow tracking server

Open a separate terminal, activate the same virtual environment, and run:

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000
```

Keep this terminal running. Open the MLflow UI at:

`http://127.0.0.1:5000`

### 2.4 Evaluate both models and compare them

In the project terminal:

```powershell
python evaluate.py
```

This evaluates Model A and Model B on `data/eval_inputs.csv`, logs their parameters, metrics and model artifacts to MLflow, and saves evaluation outputs under `outputs/`.

### 2.5 Register the Champion and Challenger aliases

```powershell
python register_champion.py
```

This registers both logged models under `sentiment-classifier` and assigns the `champion` and `challenger` aliases according to the decision rule below.

### 2.6 Predict on new data using the Champion alias

```powershell
python predict_from_mlflow.py
```

The script loads `models:/sentiment-classifier@champion`, predicts the 10 examples in `data/new_data.csv`, prints the predictions, and saves them to `outputs/new_data_predictions.csv`.

## 3. Champion Selection Rule

The rule was defined before the final evaluation run and was not changed after inspecting the results.

- **Primary metric:** Accuracy; higher is better.
- **Minimum improvement margin:** The Challenger must improve accuracy by at least **0.01 (one percentage point)** to replace the current Champion.
- **Tie-breaker:** If the accuracy difference is below 0.01, the model with lower average inference latency is preferred.
- **Latency direction:** Lower average inference latency is better.

## 4. Results

Both models were evaluated on the same 100 labelled examples using CPU inference.

| Metric | Model A: DistilBERT | Model B: TextAttack BERT | Difference (B − A) |
|---|---:|---:|---:|
| Accuracy | 0.9700 | 0.9900 | +0.0200 |
| Precision | 0.9796 | 1.0000 | +0.0204 |
| Recall | 0.9600 | 0.9800 | +0.0200 |
| F1-score | 0.9697 | 0.9899 | +0.0202 |
| Average latency (ms) | 12.8654 | 26.8266 | +13.9612 |
| Total inference time (s) | 1.2865 | 2.6827 | +1.3962 |

**Winner:** Model B — `textattack/bert-base-uncased-SST-2`.

The Challenger's accuracy improvement is 0.02, which is greater than the required 0.01 margin.

## 5. Decision

Model B, `textattack/bert-base-uncased-SST-2`, is selected as the Champion. It achieved an accuracy of 0.99, compared with 0.97 for Model A, `distilbert-base-uncased-finetuned-sst-2-english`. The absolute accuracy improvement is 0.02, or two percentage points, so Model B exceeds the pre-defined minimum improvement margin of 0.01. Model B also produced higher precision, recall and F1-score on this evaluation set.

There is a latency trade-off. Model B required approximately 26.83 ms per example, while Model A required approximately 12.87 ms. Therefore, Model A was about twice as fast in this CPU evaluation. Under the decision rule, the accuracy gain takes priority when it reaches the required margin, so the higher-latency Model B replaces Model A.

The result should be interpreted cautiously. The evaluation contains only 100 examples and uses a custom-created dataset rather than the official SST-2 evaluation split. A two-percentage-point difference on a sample of this size may not generalise to other data. The models may also behave differently on text from domains unlike their fine-tuning data. A larger, representative, independently labelled test set and repeated latency measurements would provide stronger evidence. The Champion and Challenger aliases were assigned in the MLflow Model Registry, and the prediction script loads the Champion through its alias rather than directly calling Hugging Face.

## 6. Limitations and Declaration

- The evaluation set contains 100 labelled examples; results may vary on a larger or different dataset.
- The dataset is custom-created and is not the official SST-2 benchmark split.
- Both selected models are SST-2 sentiment classifiers; performance may be weaker on text from domains unlike their training data.
- Latency was measured on CPU in the local environment and may differ on other hardware or under different system load.
- A model's prediction can be wrong even when its overall evaluation metrics are high. Individual predictions should not be treated as guaranteed ground truth.
- **Tools used:** Python, Hugging Face Transformers, PyTorch, MLflow, pandas, scikit-learn, matplotlib, Git and GitHub. ChatGPT (AI assistant) was used to help draft and debug code and documentation; the scripts were run and the results/screenshots were checked in the local project environment.
