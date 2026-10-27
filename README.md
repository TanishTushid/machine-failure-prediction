# Predictive Maintenance & Failure Prediction

An end-to-end binary classification project that estimates whether industrial
equipment is likely to fail from its type and sensor readings. It uses the
AI4I 2020 Predictive Maintenance dataset, which is **synthetic**. Its results
are educational and do not demonstrate real-world industrial performance.
Model scores are not calibrated field failure probabilities.

## Project overview

The workflow is designed around the rare-failure setting: a missed failure may
be costly, but excessive false alarms are also undesirable. Candidate models
are compared with stratified cross-validation using average precision. The
decision threshold is chosen on a validation split to maximize F2 (recall is
weighted more heavily than precision); ties select the higher threshold to
reduce false alarms. The test split is reserved for one final evaluation.

### Architecture

```text
CSV -> schema validation -> stratified train/validation/test split
    -> leakage-safe feature engineering + fitted preprocessing pipeline
    -> CV model comparison / limited randomized tuning
    -> validation threshold selection -> final test metrics
    -> trusted local model + metadata -> FastAPI /predict
```

Engineered predictors are process-minus-air temperature and a rotational
speed-times-torque proxy. They have physical motivation but remain proxies,
not measured physical quantities or causal explanations.

## Dataset

Get the synthetic dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
Place `ai4i2020.csv` at `dataset/ai4i2020.csv`; details and the configurable
path option are in [data/README.md](data/README.md). Dataset files are ignored
by Git and should not be committed.

The inspected local dataset has 10,000 rows, 14 columns, 339 failures (3.39%),
no missing values or exact duplicate rows, and machine types `L`, `M`, and
`H`. These are dataset observations only. Run [the EDA notebook](notebooks/01_eda.ipynb)
to regenerate plots under `artifacts/eda/`.

### Leakage prevention

`Machine failure` is the target. Model inputs are only `Type`, air
temperature, process temperature, rotational speed, torque, and tool wear.
`UDI` and `Product ID` are identifiers and are excluded. `TWF`, `HDF`, `PWF`,
`OSF`, and `RNF` indicate failure modes and are excluded because they can
directly reveal target-related information. Preprocessing is inside the
scikit-learn pipeline, so imputers, scaling, and category encoding are fitted
within training folds rather than on validation or test data.

## Repository structure

```text
data/README.md                   Dataset download and limitations
notebooks/01_eda.ipynb           Executable exploratory analysis
src/data_loader.py               CSV loading and schema validation
src/preprocessing.py             Feature engineering and sklearn pipeline
src/train.py                     Baselines, CV, tuning, threshold, artifacts
src/evaluate.py                  Failure-sensitive evaluation metrics
src/predict.py                   Safe-schema inference and artifact loading
src/explain.py                   SHAP global and single-row explanations
app/main.py                      FastAPI health, model-info, predict endpoints
tests/                           Preprocessing, training, and API checks
models/                          Locally generated model artifacts (not committed)
artifacts/                        EDA, evaluation, and SHAP figures/reports
```

## Installation and training

Python 3.10+ is recommended. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.train
```

Set `PREDICTIVE_MAINTENANCE_DATA` or pass `--data` if the CSV is elsewhere:

```powershell
$env:PREDICTIVE_MAINTENANCE_DATA = "C:\data\ai4i2020.csv"
python -m src.train
```

Use `python -m src.train --no-tune` for a faster baseline run or
`python -m src.train --no-mlflow` to skip tracking. Training writes the
complete preprocessing-plus-model pipeline and metadata to `models/`, model
comparison and test metrics/plots to `artifacts/`, and an MLflow run to the
local `mlruns/` directory when MLflow is enabled. These generated model files
are intentionally ignored by Git.

## Model selection and measured results

The following values are from a single run on the local synthetic dataset
(seed 42). The leaderboard is **3-fold stratified training CV average
precision**; it is not test performance. The table shows the initial baseline
models and the randomized tuned tree ensembles.

| Candidate | CV average precision |
|---|---:|
| Random Forest (tuned) | 0.8429 |
| Random Forest (baseline) | 0.8383 |
| XGBoost (baseline) | 0.8373 |
| XGBoost (tuned) | 0.8362 |
| Decision Tree | 0.7010 |
| Logistic Regression | 0.4952 |
| Dummy prior baseline | 0.0338 |

Tuning was deliberately limited to four randomized configurations per
ensemble. In this run, tuning improved the Random Forest CV score slightly,
while tuned XGBoost did not outperform its baseline. This small difference
does not establish robust improvement; results can vary across splits and
synthetic datasets.

The selected tuned Random Forest used validation F2 to choose a threshold of
approximately `0.7114`. On the untouched test split, the observed results
were:

| Precision | Recall | F1 | Average precision (PR-AUC) | ROC-AUC | False positives | False negatives |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8088 | 0.8088 | 0.8088 | 0.8584 | 0.9686 | 13 | 13 |

The test set contained 68 failures and 1,932 non-failures. The test confusion
matrix is 1,919 true negatives, 13 false positives, 13 false negatives, and
55 true positives. A false negative is a missed failure; a false positive
can trigger an unnecessary inspection. The F2 operating point reflects a
recall-weighted validation criterion, not a universal industrial cost
function. The final model and threshold are saved in `models/metadata.json`
after training; metrics and plots are reproducible with the command above.

## Explainability

After training, create model-aligned SHAP plots:

```powershell
python -m src.explain
```

The global bar plot and one example waterfall plot are saved under
`artifacts/shap/`. They describe how features influence this model's output
for this dataset. SHAP attributions are associations with the fitted model,
not causal explanations or proof of real-world behavior.

## FastAPI

Train the model first, then run:

```powershell
uvicorn app.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.
Endpoints are `GET /health`, `GET /model-info`, and `POST /predict`. Example:

```powershell
$body = @{
  Type = "M"
  air_temperature_k = 298.1
  process_temperature_k = 308.6
  rotational_speed_rpm = 1551
  torque_nm = 42.8
  tool_wear_min = 0
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" -Body $body
```

The response contains a binary class, model score, risk category based on the
saved validation threshold, and model version. The score is **not calibrated**
as a real-world failure probability. Inputs reject missing/extra fields,
unsupported machine types, non-finite numbers, and non-positive temperatures
or rotational speeds. The service loads local joblib artifacts; joblib uses
pickle internally, so **only load model files from trusted sources**.

## Docker

Training artifacts are local and excluded from Git. Generate them before
building the image:

```powershell
python -m src.train
docker build -t predictive-maintenance-api .
docker run --rm -p 8000:8000 predictive-maintenance-api
```

The image includes the generated `models/` folder from the build context.
Do not copy or publish an image containing artifacts you do not trust.

## Tests and reproducibility

```powershell
python -m pytest -q
```

The GitHub Actions workflow runs the test suite on pushes and pull requests.
Splits and estimators use seed `42`; CV is stratified. Re-running the full
workflow makes a new test evaluation for reproducibility checks, so avoid
repeatedly selecting models or thresholds based on those test results.

## Limitations and future work

- The data is synthetic; it does not capture all real sensor noise, operating
  regimes, maintenance interventions, or changing failure rates.
- The output score has not been calibrated and should not drive operational
  decisions.
- One seeded split and a compact search do not establish production
  reliability; prospective, time-aware validation on representative data is
  needed.
- The illustrative F2 criterion is not a substitute for an organization’s
  cost-based alert policy.
- Future improvements include probability calibration on separate data,
  drift monitoring, time-aware validation, and deployment against approved
  real-world data with domain experts.
