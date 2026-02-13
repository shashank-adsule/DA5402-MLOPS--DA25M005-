import pandas as pd
import numpy as np
import yaml
import json
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import subprocess
import re

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_git_commit_hash():
    """Get current git commit hash for reproducibility"""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
        return commit_hash
    except:
        return "no_git_repo"


def log_to_registry(message, registry_path='models/model_metadata.log'):
    """Log model training events to registry"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    with open(registry_path, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")


def load_training_data(config):
    """Load processed training data"""
    version = config['versioning']['data_version']
    data_path = os.path.join(
        config['data']['processed_dir'],
        f"{version}_train_processed.csv"
    )
    
    print(f"Loading training data from: \u001b[32m{data_path}\u001b[0m")
    
    try:
        df = pd.read_csv(data_path)
        log_to_registry(f"Loaded training data: {data_path} ({len(df)} samples)")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Please run data_prep.py first."
        )


def prepare_features_target(df, config):
    """Separate features and target variable"""
    target_col = config['features']['target']
    
    # Exclude non-feature columns
    exclude_cols = ['UDI', 'Product ID', *config['features']['targets'], 'Failure Type']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target: {target_col}")
    print(f"  - Class 0 (No Failure): {(y == 0).sum()} samples")
    print(f"  - Class 1 (Failure): {(y == 1).sum()} samples")
    
    log_to_registry(f"Features: {len(feature_cols)}, Target: {target_col}")
    
    return X, y, feature_cols


def train_model(X_train, y_train, config):
    """Train the Random Forest classifier"""
    print("\nTraining Random Forest model...")
    
    # Get hyperparameters from config
    params = config['model_params']
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
        class_weight=params['class_weight'],
        n_jobs=params['n_jobs']
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    print(f"✓ Model trained successfully")
    print(f"  - Algorithm: {params['algorithm']}")
    print(f"  - n_estimators: {params['n_estimators']}")
    print(f"  - max_depth: {params['max_depth']}")
    
    log_to_registry(
        f"Model trained: {params['algorithm']} with "
        f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']}"
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\nModel Performance Metrics:")
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    log_to_registry(
        f"Evaluation metrics - Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
    )
    
    return metrics, cm


def get_feature_importance(model, feature_names):
    """Get and display feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:30s} {row['importance']:.4f}")
    
    return importance_df

def get_next_model_version(folder_path):
    pattern = r"model_v(\d+)\.pkl"
    versions = []

    for file in os.listdir(folder_path):
        match = re.match(pattern, file)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        return 1   # If no model exists, start from v1

    return max(versions) + 1

def save_model(model, config):
    """Save trained model to disk"""
    model_path = config['deployment']['model_path']
    
    version=f"v{get_next_model_version(os.path.dirname(model_path))}"

    # print(version)


    # Create directory if needed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    


    # Save model
    with open(f"{model_path}{version}.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Model saved to: {model_path}")
    log_to_registry(f"Model saved: {model_path}")
    
    return model_path, version


def save_metadata(metrics, feature_names, importance_df, config, version):
    """Save model metadata for reproducibility"""
    metadata_path = config['deployment']['metadata_path']
    
    # Get git commit hash
    commit_hash = get_git_commit_hash()
    
    # Create metadata dictionary
    metadata = {
        'model_version': version,
        'data_version': config['versioning']['data_version'],
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'git_commit': commit_hash,
        'config_file': 'config.yaml',
        'algorithm': config['model_params']['algorithm'],
        'hyperparameters': config['model_params'],
        'metrics': {k: float(v) for k, v in metrics.items()},
        'features': feature_names,
        'feature_importance': importance_df.to_dict('records')[:10],
        'data_path': os.path.join(
            config['data']['processed_dir'],
            f"{config['versioning']['data_version']}_train_processed.csv"
        )
    }
    
    # Save as JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    log_to_registry(f"Metadata saved: {metadata_path}")
    
    # Also append to model registry log
    registry_path = config['logging']['model_registry']
    with open(registry_path, 'a') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"MODEL VERSION: {metadata['model_version']}\n")
        f.write(f"Training Date: {metadata['training_date']}\n")
        f.write(f"Data Version: {metadata['data_version']}\n")
        f.write(f"Git Commit: {commit_hash}\n")
        f.write(f"Algorithm: {metadata['algorithm']}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write("=" * 70 + "\n")
    
    return metadata


def main():
    """Main training pipeline"""
    print(f"\u001b[33m{'MODEL TRAINING PIPELINE':=^75}\u001b[0m")
    
    # Load configuration
    config = load_config()
    print(f"\n✓ Loaded configuration from config.yaml")
    
    # Initialize model registry log
    registry_path = config['logging']['model_registry']
    if not os.path.exists(registry_path):
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL REGISTRY LOG\n")
            f.write("=" * 70 + "\n\n")
    
    # Step 1: Load training data
    print("\n[STEP 1] Loading training data...")
    df = load_training_data(config)
    
    # Step 2: Prepare features and target
    print("\n[STEP 2] Preparing features and target...")
    X, y, feature_names = prepare_features_target(df, config)
    
    # Step 3: Split into train/validation sets
    print("\n[STEP 3] Splitting into train/validation sets...")
    test_size = config['training']['test_split']
    shuffle = config['training']['shuffle']
    
    # Note: stratify requires shuffle=True
    if shuffle and config['training']['stratify']:
        stratify = y
    else:
        stratify = None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=config['model_params']['random_state'],
        stratify=stratify,
        shuffle=shuffle
    )
    
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Validation set: {len(X_val)} samples")
    
    # Step 4: Train model
    print("\n[STEP 4] Training model...")
    model = train_model(X_train, y_train, config)
    
    # Step 5: Evaluate model
    print("\n[STEP 5] Evaluating model...")
    metrics, cm = evaluate_model(model, X_val, y_val)
    
    # Step 6: Feature importance
    print("\n[STEP 6] Analyzing feature importance...")
    importance_df = get_feature_importance(model, feature_names)
    
    # Step 7: Save model
    print("\n[STEP 7] Saving model artifacts...")
    model_path, version = save_model(model, config)
    
    # Step 8: Save metadata
    print("\n[STEP 8] Saving metadata...")
    metadata = save_metadata(metrics, feature_names, importance_df, config,version)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {config['deployment']['metadata_path']}")
    print(f"Model version: {metadata['model_version']}")
    print(f"Data version: {metadata['data_version']}")
    print(f"Git commit: {metadata['git_commit']}")
    print("\nNext steps:")
    print("  1. Review model performance metrics")
    print("  2. Deploy model using inference.py")
    print("  3. Monitor production performance using monitor.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
