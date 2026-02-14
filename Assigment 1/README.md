# Manual MLOps Project: Predictive Maintenance System

## Overview
This project implements a complete MLOps pipeline for a Predictive Maintenance System using **manual methods only** - no MLflow, DVC, Airflow, or Kubernetes. The goal is to understand the pain points of manual ML management and appreciate the value of automation.

## Project Structure
```
manual_mlops_project/
│
├── data/
│   ├── raw/                    # Immutable raw data
│   │   └── ai4i2020.csv
│   ├── processed/              # Versioned processed data
│   │   ├── v0_raw.csv
│   │   └── v1_train_processed.csv
│   ├── production/             # Production monitoring data
│   │   ├── v1_production.csv
│   │   └── predictions.csv
│   └── manifest.txt            # Data lineage log
│
├── models/
│   ├── model_v#.pkl            # Trained models (#-> number of version)
│   ├── model_metadata.json     # Model metadata
│   ├── model_metadata.log      # Model registry log
│   └── monitoring_log.csv      # Production monitoring log
│
├── src/
│   ├── data_prepration.py            # Data preparation pipeline
│   ├── train.py                # Model training script
│   ├── create_plots.py                # create traing plots for the infence of models
│   ├── inference.py            # Flask API for predictions
│   └── monitor.py              # Production monitoring
│
├── api/
│   └──test_api.py                 # API smoke tests 
│
├── logs/
│   ├── monitoring_report.txt      # Production monitoring log
│   └── deployment_log.csv          # Deployment history 
│
├── config.yaml                 # Single source of truth for configuration
├── AI_DISCLOUSE                 # Single source of truth for configuration
├── Report.pdf                 # 2-3 page Documentaion Report for the challenges and issuses faced during this assigment 
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
download dataset from [DATASET](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

### 3. Run the Complete Pipeline

#### Phase A: Data Preparation
```bash
python src/data_prep.py
```
- Loads raw data
- Cleans and engineers features
- Splits chronologically (7000 train, 3000 production)
- Saves versioned datasets
- Logs all steps to `data/manifest.txt`

#### Phase B: Model Training
```bash
python src/train.py
```
- Trains Random Forest classifier
- Evaluates performance
- Saves model and metadata
- Logs to model registry

#### Phase C: Deploy API
```bash
python src/inference.py
```
API runs on `http://127.0.0.1:5000` with endpoints:
- `GET /health` - Health check
- `GET /model-info` - Model metadata
- `POST /predict` - Single prediction
- `POST /batch-predict` - Batch predictions

**In a separate terminal**, run tests:
```bash
python test_api.py
```

#### Phase D: Production Monitoring
```bash
python src/monitor.py
```
- Evaluates production data
- Detects feature drift
- Compares with training baseline
- Triggers retraining if needed

## Configuration

All settings are in `config.yaml`:

```yaml
# Data paths
data:
  raw_path: "data/raw/ai4i2020.csv"
  current_version: "v1"
  train_size: 7000

# Model hyperparameters
model_params:
  algorithm: "RandomForest"
  n_estimators: 100
  max_depth: 10
  random_state: 42

# Monitoring thresholds
monitoring:
  accuracy_threshold: 0.80
  precision_threshold: 0.75
  drift_threshold: 0.15
```

To change hyperparameters, edit `config.yaml` and retrain:
```bash
# Edit config.yaml (e.g., change n_estimators to 200)
python src/train.py
```

## Data Versioning

The project uses a **manual data versioning system**:

1. **Raw Data**: Stored in `data/raw/` (immutable)
2. **Processed Data**: Versioned in `data/processed/` as `v1_train_processed.csv`, etc.
3. **Manifest File**: `data/manifest.txt` logs which script produced which version
4. **Example**:
   ```
   [2026-02-09 10:30:45] Loading raw data from data/raw/ai4i2020.csv
   [2026-02-09 10:30:45] Successfully loaded 10000 samples
   [2026-02-09 10:30:46] Data cleaning complete: 10000 -> 10000 rows
   [2026-02-09 10:30:46] Saved processed data version v1: data/processed/v1_train_processed.csv
   ```

## Model Registry

The **manual model registry** tracks:

1. **Model File**: `models/model_v1.pkl` (pickled sklearn model)
2. **Metadata**: `models/model_metadata.json` containing:
   - Model version
   - Data version used
   - Training date
   - Git commit hash
   - Hyperparameters
   - Performance metrics
   - Feature importance

3. **Registry Log**: `models/model_metadata.log` - human-readable history

## Deployment Tracking

`deployment_log.csv` tracks all API deployments:

| timestamp           | model_version | model_path         | status   | port |
|---------------------|---------------|--------------------|----------|------|
| 2026-02-09 10:35:00 | v1            | models/model_v1.pkl| deployed | 5000 |

## Reproducibility

To reproduce model training:

1. Check `models/model_metadata.json` for:
   - Data version used (e.g., `v1`)
   - Git commit hash
   - Hyperparameters

2. Load the same data version:
   ```python
   df = pd.read_csv('data/processed/v1_train_processed.csv')
   ```

3. Use the same config:
   ```bash
   python src/train.py
   ```

## Testing

Three smoke tests verify API functionality:

```bash
python api/test_api.py
```

Tests:
1. **Health Check**: Verify API is running
2. **Single Prediction**: Test prediction format
3. **Edge Case**: Test high tool wear scenario

## Monitoring & Drift Detection

`src/monitor.py` performs:

1. **Performance Monitoring**: Calculates metrics on production data
2. **Drift Detection**: Compares feature distributions
3. **Retraining Triggers**: Flags when metrics drop below thresholds

Output: `monitoring_report.txt` with recommendations

## Simulating Time-Series Drift

The dataset includes built-in drift:
- **Training**: First 7000 samples (normal conditions)
- **Production**: Last 3000 samples (aged equipment, increased tool wear)

This simulates real-world model degradation over time.

## Example API Usage

### Single Prediction
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Air temperature [K]": 298.5,
    "Process temperature [K]": 308.7,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 50,
    "Type": "M"
  }'
```

Response:
```json
{
  "prediction": 0,
  "failure_probability": 0.15,
  "prediction_class": "No Failure",
  "model_version": "v1",
  "timestamp": "2026-02-09T10:40:00"
}
```

## Retraining Workflow

If monitoring detects degradation:

1. Review `monitoring_report.txt`
2. Investigate root causes
3. Update `config.yaml` if needed (e.g., new thresholds)
4. Retrain: `python src/train.py`
5. Update model version in `config.yaml`
6. Restart API: `python src/inference.py`
7. Verify with tests: `python test_api.py`

## Key Pain Points (Manual MLOps)

This project highlights:

1. **Data Versioning**: Manual manifest files are error-prone
2. **Model Registry**: No automatic tracking of experiments
3. **Reproducibility**: Git hash tracking is manual
4. **Deployment**: Manual logging of which model is live
5. **Monitoring**: No automatic alerts or dashboards
6. **Retraining**: Manual trigger checking and execution

These pain points motivate automated MLOps tools!

## Assignment Requirements Checklist

- [x] Manual data versioning with manifest.txt
- [x] Configuration isolated in config.yaml
- [x] Model registry with metadata.json
- [x] Git commit hash tracking
- [x] Flask API with deployment logging
- [x] Three smoke tests
- [x] Production monitoring
- [x] Drift detection
- [x] Retraining trigger logic
- [x] Complete documentation

## Author
Shashank Satish Adsule - DA5402 MLOps Course
