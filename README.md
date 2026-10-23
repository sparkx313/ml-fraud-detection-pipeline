# Transaction Fraud Detection Pipeline

An offline, CSV-driven workflow for training and applying a binary transaction fraud classifier. Bring labeled transaction data; this repository does not bundle fabricated records or claim pre-measured performance.

## Train and evaluate

Install Python 3.11 or newer and the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

The training CSV must include a numeric binary target column: `0` for legitimate and `1` for fraud. All remaining columns are treated as features unless you designate a transaction identifier with `--id-column`.

```bash
python fraud_detector.py train --data path/to/labeled_transactions.csv --target is_fraud --id-column transaction_id --test-size 0.2 --threshold 0.5
```

The command uses a stratified random holdout and a preprocessing pipeline with median imputation and scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical columns, and class-weighted logistic regression. It writes holdout predictions, a JSON report, and a reusable `joblib` model artifact. The report includes precision, recall, F1, balanced accuracy, average precision, ROC AUC, and the confusion matrix. Change the threshold to examine a different precision/recall tradeoff.

Default output files are `fraud_predictions.csv`, `fraud_metrics.json`, and `fraud_model.joblib`. Provide `--predictions`, `--metrics`, and `--model` to choose other paths. `--seed` sets the split seed.

## Score new transactions

Use the saved model with a CSV containing the feature columns used during training:

```bash
python fraud_detector.py predict --model fraud_model.joblib --input path/to/new_transactions.csv --output fraud_scores.csv
```

The output includes each source row, the fraud probability, and the thresholded binary prediction. If an ID column was configured during training and is present in the new CSV, it is copied into the output. `--threshold` can override the threshold saved in the model.

## Docker

Build the image:

```bash
docker build -t transaction-fraud-pipeline .
```

Mount a host directory containing your data and artifacts:

```bash
docker run --rm -v /absolute/path/to/data:/data transaction-fraud-pipeline train --data /data/labeled_transactions.csv --target is_fraud --id-column transaction_id --model /data/fraud_model.joblib --predictions /data/fraud_predictions.csv --metrics /data/fraud_metrics.json
```

The workflow reads and writes local files and makes no network requests.

## Scope and evaluation limits

This is a reproducible baseline, not a real-time transaction service. Its default random holdout can overstate performance when transactions from the same account or time period appear on both sides of the split; use account-grouped or time-based evaluation when that matches the deployment setting. The threshold and reported metrics are not a guarantee of fraud detection quality. Validate the model against the costs of false positives and missed fraud before operational use.

## License

MIT. See [LICENSE](LICENSE).
