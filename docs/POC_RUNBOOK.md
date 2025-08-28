# POC Runbook

Prerequisites
- Python 3.8+ installed.
- Internet access for the Kaggle API (required to download dataset).
- `kaggle.json` credential file placed in the repository root (the repo already contains this; it's listed in `.gitignore`).

1) Prepare Kaggle credentials
Run these commands from the repository root (do them separately):
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

2) Create and activate a Python virtual environment, then install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3) Verify Kaggle CLI is available
kaggle --version

4) Download competition data (will unzip into ./data)
kaggle competitions download -c playground-series-s5e5 -p ./data --unzip

If the above does not auto-unzip, run:
kaggle competitions download -c playground-series-s5e5 -p ./data
unzip ./data/playground-series-s5e5.zip -d ./data

5) Verify required files exist
ls -la ./data/train.csv ./data/test.csv ./data/sample_submission.csv

6) Create a subsample for fast POC (default 10000 rows)
python3 scripts/subsample.py --input ./data/train.csv --output ./data/subsample_train.csv --n 10000

7) Train a quick POC model (XGBoost)
python3 scripts/train.py --train ./data/subsample_train.csv --model_out ./models/poc_model.pkl

The script will save metrics to ./models/poc_model_metrics.json alongside the model.

8) Produce predictions in sample-submission format
python3 scripts/predict.py --model ./models/poc_model.pkl --test ./data/test.csv --out ./submissions/predictions.csv

9) Quick validation of submission format
head -n 1 ./data/sample_submission.csv
head -n 5 ./submissions/predictions.csv

10) Prepare final submission for human upload (do not submit automatically)
mv ./submissions/predictions.csv ./submissions/final_submission.csv

Notes and cautions
- The repository contains `kaggle.json` and it is included in `.gitignore`; do not commit credentials to version control.
- The README forbids accessing the Kaggle website in a browser; using the Kaggle API/CLI programmatically is allowed per the run plan approved by the human.
- If you need a smaller subsample, reduce `--n` when calling `scripts/subsample.py`.
- All long-running or heavy training should be done with consideration of available CPU/RAM; this POC uses small defaults.