"""
Monitoring Script - Phase D
Monitor production performance, detect drift, trigger retraining
"""

import pandas as pd
import numpy as np
import yaml
import json
import pickle
import os, re
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import csv


def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_path):
    """Load trained model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_metadata(metadata_path):
    """Load model metadata"""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_production_data(config):
    """Load production data for monitoring"""
    version = config['versioning']['data_version']
    prod_path = os.path.join(
        config['data']['production_dir'],
        f"{version}_production.csv"
    )
    
    if not os.path.exists(prod_path):
        print(f"✗ Production data not found at {prod_path}")
        print("Please run data_prep.py first.")
        return None
    
    df = pd.read_csv(prod_path)
    print(f"✓ Loaded {len(df)} production samples from {prod_path}")
    return df


def calculate_production_metrics(model, X_prod, y_prod):
    """Calculate metrics on production data"""
    # Make predictions
    y_pred = model.predict(X_prod)
    y_pred_proba = model.predict_proba(X_prod)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_prod, y_pred),
        'precision': precision_score(y_prod, y_pred, zero_division=0),
        'recall': recall_score(y_prod, y_pred, zero_division=0),
        'f1_score': f1_score(y_prod, y_pred, zero_division=0),
        'samples': len(y_prod)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_prod, y_pred)
    
    return metrics, cm, y_pred


def compare_with_training_metrics(prod_metrics, training_metrics):
    """Compare production metrics with training baseline"""
    print("\n" + "=" * 70)
    print("METRICS COMPARISON: Production vs Training")
    print("=" * 70)
    
    comparison = {}
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        train_val = training_metrics.get(metric, 0)
        prod_val = prod_metrics.get(metric, 0)
        diff = prod_val - train_val
        pct_change = (diff / train_val * 100) if train_val > 0 else 0
        
        comparison[metric] = {
            'training': train_val,
            'production': prod_val,
            'difference': diff,
            'pct_change': pct_change
        }
        
        status = "✓" if diff >= 0 else "⚠"
        print(f"{metric.capitalize():12s} | Training: {train_val:.4f} | "
              f"Production: {prod_val:.4f} | "
              f"Change: {pct_change:+.2f}% {status}")
    
    print("=" * 70)
    
    return comparison


def detect_feature_drift(df_train, df_prod,config, threshold=0.15, ):
    """
    Detect distribution drift in features
    Using simple statistical comparison (mean and std deviation)
    """
    print(f"\u001b[33m{'FEATURE DRIFT DETECTION':=^75}\u001b[0m")

    # Exclude non-numeric and target columns
    exclude_cols = ['UDI', 'Product ID', config["features"]["targets"], 'Failure Type']
    numeric_cols = [col for col in df_train.columns 
                    if col not in exclude_cols and 
                    pd.api.types.is_numeric_dtype(df_train[col])]
    
    drift_detected = False
    drift_features = []
    
    for col in numeric_cols:
        # Calculate statistics
        train_mean = df_train[col].mean()
        train_std = df_train[col].std()
        
        prod_mean = df_prod[col].mean()
        prod_std = df_prod[col].std()
        
        # Calculate normalized difference
        mean_diff = abs(prod_mean - train_mean) / (train_std + 1e-8)
        std_diff = abs(prod_std - train_std) / (train_std + 1e-8)
        
        # Check if drift exceeds threshold
        if mean_diff > threshold or std_diff > threshold:
            drift_detected = True
            drift_features.append({
                'feature': col,
                'train_mean': train_mean,
                'prod_mean': prod_mean,
                'mean_drift': mean_diff,
                'std_drift': std_diff
            })
            
            print(f"⚠ DRIFT DETECTED in '{col}':")
            print(f"    Mean: {train_mean:.4f} → {prod_mean:.4f} "
                  f"(drift: {mean_diff:.4f})")
            print(f"    Std:  {train_std:.4f} → {prod_std:.4f} "
                  f"(drift: {std_diff:.4f})")
    
    if not drift_detected:
        print("✓ No significant drift detected in features")
    else:
        print(f"\n⚠ Total features with drift: {len(drift_features)}")
    
    print("=" * 70)
    
    return drift_detected, drift_features


def check_retraining_trigger(prod_metrics, config):
    """
    Check if retraining should be triggered based on thresholds
    """
    print(f"\u001b[33m{'RETRAINING TRIGGER CHECK':=^75}\u001b[0m")
    
    thresholds = config['monitoring']
    
    triggers = []
    
    # Check accuracy
    if prod_metrics['accuracy'] < thresholds['accuracy_threshold']:
        triggers.append(
            f"Accuracy ({prod_metrics['accuracy']:.4f}) below threshold "
            f"({thresholds['accuracy_threshold']:.4f})"
        )
    
    # Check precision
    if prod_metrics['precision'] < thresholds['precision_threshold']:
        triggers.append(
            f"Precision ({prod_metrics['precision']:.4f}) below threshold "
            f"({thresholds['precision_threshold']:.4f})"
        )
    
    # Check recall
    if prod_metrics['recall'] < thresholds['recall_threshold']:
        triggers.append(
            f"Recall ({prod_metrics['recall']:.4f}) below threshold "
            f"({thresholds['recall_threshold']:.4f})"
        )
    
    if triggers:
        print("⚠ RETRAINING RECOMMENDED - Triggers:")
        for i, trigger in enumerate(triggers, 1):
            print(f"  {i}. {trigger}")
    else:
        print("✓ All metrics above thresholds - No retraining needed")
    
    print("=" * 70)
    
    return len(triggers) > 0, triggers


def log_monitoring_results(prod_metrics, drift_detected, triggers, config):
    """Log monitoring results to file"""
    log_path = config['logging']['monitoring_log']
    
    # Create directory if needed
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.isfile(log_path)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_version = f"v{get_next_model_version(os.path.dirname(config['deployment']['model_path']))}"
    
    # Write to CSV
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'timestamp', 'model_version', 'samples', 'accuracy', 
                'precision', 'recall', 'f1_score', 'drift_detected',
                'retraining_needed', 'trigger_reasons'
            ])
        
        # Write monitoring record
        writer.writerow([
            timestamp,
            model_version,
            prod_metrics['samples'],
            f"{prod_metrics['accuracy']:.4f}",
            f"{prod_metrics['precision']:.4f}",
            f"{prod_metrics['recall']:.4f}",
            f"{prod_metrics['f1_score']:.4f}",
            drift_detected,
            len(triggers) > 0,
            '; '.join(triggers) if triggers else 'None'
        ])
    
    print(f"\n✓ Monitoring results logged to {log_path}")


def generate_monitoring_report(prod_metrics, comparison, drift_features, 
                               triggers, config):
    """Generate a detailed monitoring report"""
    report_path = config["logging"]["monitoring_report"]
    os.makedirs(os.path.dirname(report_path),exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PRODUCTION MONITORING REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Version: {config['versioning']['model_version']}\n")
        f.write(f"Data Version: {config['versioning']['data_version']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("PRODUCTION METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Samples Evaluated: {prod_metrics['samples']}\n")
        f.write(f"Accuracy:  {prod_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {prod_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {prod_metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {prod_metrics['f1_score']:.4f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("COMPARISON WITH TRAINING BASELINE\n")
        f.write("-" * 70 + "\n")
        for metric, values in comparison.items():
            f.write(f"{metric.capitalize()}:\n")
            f.write(f"  Training:   {values['training']:.4f}\n")
            f.write(f"  Production: {values['production']:.4f}\n")
            f.write(f"  Change:     {values['pct_change']:+.2f}%\n\n")
        
        if drift_features:
            f.write("-" * 70 + "\n")
            f.write("DRIFT DETECTION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Features with Drift: {len(drift_features)}\n\n")
            for drift in drift_features:
                f.write(f"Feature: {drift['feature']}\n")
                f.write(f"  Mean Drift: {drift['mean_drift']:.4f}\n")
                f.write(f"  Std Drift:  {drift['std_drift']:.4f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("RETRAINING RECOMMENDATION\n")
        f.write("-" * 70 + "\n")
        if triggers:
            f.write("⚠ RETRAINING RECOMMENDED\n\n")
            f.write("Reasons:\n")
            for i, trigger in enumerate(triggers, 1):
                f.write(f"{i}. {trigger}\n")
        else:
            f.write("No retraining needed at this time\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Detailed report saved to {report_path}")

def get_next_model_version(folder_path):
    pattern = r"model_v(\d+)\.pkl"
    versions = []

    for file in os.listdir(folder_path):
        match = re.match(pattern, file)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        return 1   # If no model exists, start from v1

    return max(versions)

def main():
    """Main monitoring pipeline"""

    print(f"\u001b[33m{'PRODUCTION MONITORING':=^75}\u001b[0m")
    
    # Load configuration
    config = load_config()
    print(f"✓ Loaded configuration")
    
    version=f"v{get_next_model_version(os.path.dirname(config['deployment']['model_path']))}"

    # Load model and metadata
    print("\n[STEP 1] Loading model and metadata...")
    model_path = f"{config['deployment']['model_path']}{version}.pkl"
    metadata_path = config['deployment']['metadata_path']
    
    model = load_model(model_path)
    metadata = load_metadata(metadata_path)
    
    print(f"✓ Model version: {metadata['model_version']}")
    print(f"✓ Training accuracy: {metadata['metrics']['accuracy']:.4f}")
    
    # Load production data
    print("\n[STEP 2] Loading production data...")
    df_prod = load_production_data(config)
    
    if df_prod is None:
        return
    
    # Prepare features and target
    print("\n[STEP 3] Preparing production features...")
    target_col = config['features']['target']
    exclude_cols = ['UDI', 'Product ID', *config['features']['targets'], 'Failure Type']
    feature_cols = [col for col in df_prod.columns if col not in exclude_cols]
    
    X_prod = df_prod[feature_cols]
    y_prod = df_prod[target_col]
    
    print(f"✓ Production samples: {len(X_prod)}")
    print(f"  - Failures: {(y_prod == 1).sum()}")
    print(f"  - No Failures: {(y_prod == 0).sum()}")
    
    # Calculate production metrics
    print("\n[STEP 4] Calculating production metrics...")
    prod_metrics, cm, y_pred = calculate_production_metrics(model, X_prod, y_prod)
    
    print(f"\nProduction Performance:")
    print(f"  Accuracy:  {prod_metrics['accuracy']:.4f}")
    print(f"  Precision: {prod_metrics['precision']:.4f}")
    print(f"  Recall:    {prod_metrics['recall']:.4f}")
    print(f"  F1-Score:  {prod_metrics['f1_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Compare with training metrics
    print("\n[STEP 5] Comparing with training baseline...")
    comparison = compare_with_training_metrics(prod_metrics, metadata['metrics'])
    
    # Load training data for drift detection
    print("\n[STEP 6] Detecting feature drift...")
    version = config['versioning']['data_version']
    train_path = os.path.join(
        config['data']['processed_dir'],
        f"{version}_train_processed.csv"
    )
    df_train = pd.read_csv(train_path)
    
    drift_detected, drift_features = detect_feature_drift(
        df_train, df_prod,config,
        threshold=config['monitoring']['drift_threshold']
    )
    
    # Check retraining triggers
    print("\n[STEP 7] Checking retraining triggers...")
    needs_retraining, triggers = check_retraining_trigger(prod_metrics, config)
    
    # Log results
    print("\n[STEP 8] Logging monitoring results...")
    log_monitoring_results(prod_metrics, drift_detected, triggers, config)
    
    # Generate detailed report
    print("\n[STEP 9] Generating monitoring report...")
    generate_monitoring_report(
        prod_metrics, comparison, drift_features, triggers, config
    )
    
    # Final recommendations
    print(f"\u001b[33m{'MONITORING SUMMARY':=^75}\u001b[0m")
    print(f"Production Accuracy: {prod_metrics['accuracy']:.4f}")
    print(f"Drift Detected: {'Yes' if drift_detected else 'No'}")
    print(f"Retraining Needed: {'Yes' if needs_retraining else 'No'}")
    
    if needs_retraining:
        print("\n⚠ RECOMMENDED ACTIONS:")
        print("  1. Review monitoring report and drift analysis")
        print("  2. Investigate root causes of performance degradation")
        print("  3. Update config.yaml if needed")
        print("  4. Retrain model: python src/train.py")
        print("  5. Redeploy API: python src/inference.py")
    else:
        print("\n✓ System is performing within acceptable parameters")
        print("  Continue normal monitoring schedule")
    
    print("=" * 70)


if __name__ == "__main__":
    main()