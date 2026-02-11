import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle
import subprocess

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def log_to_manifest(message, manifest_path='data/manifest.txt', time=True):
    """Log data processing steps to manifest file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(manifest_path, 'a') as f:
        if time:
            f.write(f"[{timestamp}] {message}\n")
        else:
            f.write(f"{message}\n")


def load_raw_data(config):
    """Load raw data from specified path in config"""
    raw_path = config['data']['raw_path']
    log_to_manifest(f"Loading raw data from {raw_path}")
    
    try:
        df = pd.read_csv(raw_path) 
        print(f"✓ Loaded {len(df)} samples from \u001b[32m{raw_path}\u001b[0m")
        log_to_manifest(f"Successfully loaded {len(df)} samples")
        return df
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        log_to_manifest(f"ERROR: {error_msg}")
        raise


def clean_data(df, config):
    """Clean the dataset - handle missing values, outliers"""
    log_to_manifest("Starting data cleaning process")
    
    initial_count = len(df)
    
    # Check for missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"⚠ Found {missing_before} missing values")
        log_to_manifest(f"Found {missing_before} missing values")
        # Drop rows with missing values
        df = df.dropna()
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠ Found {duplicates} duplicate rows")
        log_to_manifest(f"Removed {duplicates} duplicate rows")
        df = df.drop_duplicates()
    
    final_count = len(df)
    removed = initial_count - final_count
    
    if removed > 0:
        print(f"✓ Cleaned data: removed {removed} rows")
        log_to_manifest(f"Data cleaning complete: {initial_count} -> {final_count} rows")
    else:
        print("✓ No cleaning needed - data is clean")
        log_to_manifest("Data is clean - no rows removed")
    
    return df


def engineer_features(df, config):
    """Create new features based on domain knowledge"""
    log_to_manifest("Starting feature engineering")
    
    df_engineered = df.copy()
    features_created = []
    
    # Create power feature (Torque * Rotational Speed)
    if config['features'].get('create_power_feature', True):
        df_engineered['Power'] = (
            df_engineered['Torque [Nm]'] * 
            df_engineered['Rotational speed [rpm]']
        )
        features_created.append('Power')
    
    # Create temperature difference
    if config['features'].get('create_temp_diff', True):
        df_engineered['Temp_diff'] = (
            df_engineered['Process temperature [K]'] - 
            df_engineered['Air temperature [K]']
        )
        features_created.append('Temp_diff')
    
    # Create interaction features
    if config['features'].get('create_interactions', True):
        df_engineered['Torque_ToolWear'] = (
            df_engineered['Torque [Nm]'] * 
            df_engineered['Tool wear [min]']
        )
        df_engineered['Speed_ToolWear'] = (
            df_engineered['Rotational speed [rpm]'] * 
            df_engineered['Tool wear [min]']
        )
        features_created.extend(['Torque_ToolWear', 'Speed_ToolWear'])
    
    print(f"✓ Created {len(features_created)} new features: {', '.join(features_created)}")
    log_to_manifest(f"Feature engineering complete: created {features_created}")
    
    return df_engineered


def split_train_production(df, config):
    train_size = config['data']['train_size']
    
    df_train = df.iloc[:train_size].copy()
    df_production = df.iloc[train_size:].copy()

    print(f"✓ Split data: \u001b[33m{len(df_train)}\u001b[0m training, \u001b[33m{len(df_production)}\u001b[0m production")
    log_to_manifest(
        f"Chronological split: {len(df_train)} training samples, "
        f"{len(df_production)} production samples"
    )
    
    return df_train, df_production


def encode_categorical(df, config, fitted_encoder=None):
    """Encode categorical variables"""
    categorical_features = config['features']['categorical_features']
    
    df_encoded = df.copy()
    
    if fitted_encoder is None:
        # Fit new encoder (training phase)
        encoder = LabelEncoder()
        for col in categorical_features:
            df_encoded[col + '_encoded'] = encoder.fit_transform(df[col])
            df_encoded = df_encoded.drop(col, axis=1)
        
        log_to_manifest(f"Encoded categorical features: {categorical_features}")
        return df_encoded, encoder
    else:
        # Use fitted encoder (production phase)
        for col in categorical_features:
            df_encoded[col + '_encoded'] = fitted_encoder.transform(df[col])
            df_encoded = df_encoded.drop(col, axis=1)
        
        return df_encoded


def scale_features(df, config, fitted_scaler=None):
    """Scale numerical features"""
    # Get all numeric columns except target and identifiers
    exclude_cols = ['UDI', 'Product ID', 'Machine failure',"TWF","HDF","PWF","OSF","RNF",'Failure Type']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    df_scaled = df.copy()
    
    if fitted_scaler is None:
        # Fit new scaler (training phase)
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        log_to_manifest(f"Scaled {len(numeric_cols)} numerical features")
        return df_scaled, scaler
    else:
        # Use fitted scaler (production phase)
        df_scaled[numeric_cols] = fitted_scaler.transform(df[numeric_cols])
        return df_scaled


def save_processed_data(df, version, config):
    """Save processed data with version number"""
    processed_dir = config['data']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    filepath = os.path.join(processed_dir, f"{version}_processed.csv")
    df.to_csv(filepath, index=False)
    
    print(f"✓ Saved processed data: \u001b[32m{filepath}\u001b[0m")
    log_to_manifest(f"Saved processed data version {version}: {filepath}")
    
    return filepath

def get_git_commit_hash():
    """Get current git commit hash for reproducibility"""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
        return commit_hash
    except:
        return "no_git_repo"

def main():
    """Main data preparation pipeline"""
    print(f"\u001b[33m{'| DATA PREPARATION PIPELINE |':=^75}\u001b[0m")
    
    # Load configuration
    config = load_config()
    print(f"\n✓ Loaded configuration from \u001b[32m./config.yaml \u001b[0m")

    os.makedirs(config["data"]["data_path"],exist_ok=True)
    
    # Initialize manifest
    manifest_path = config['logging']['manifest_file']
    with open(manifest_path, 'w') as f:
        f.write(f"{'| DATA PROCESSING MANIFEST |':=^75}\n\n")
    
    # Step 1: Load raw data
    print("\n[STEP 1] Loading raw data...")
    df_raw = load_raw_data(config)
    
    df_raw.to_csv(os.path.join(config['data']['processed_dir'],"v0_raw.csv"))
    
    # Step 2: Clean data
    print("\n[STEP 2] Cleaning data...")
    df_cleaned = clean_data(df_raw, config)

     # Step 3: Split train/production chronologically
    print("\n[STEP 3] Splitting data chronologically...")
    df_train, df_production = split_train_production(df_cleaned, config)
    
    # Step 4: Feature engineering (on training data first)
    print("\n[STEP 4] Engineering features...")
    df_train_engineered = engineer_features(df_train, config)
    df_production_engineered = engineer_features(df_production, config)
    
    # Step 5: Encode categorical variables
    print("\n[STEP 5] Encoding categorical features...")
    df_train_encoded, encoder = encode_categorical(df_train_engineered, config)
    df_production_encoded = encode_categorical(df_production_engineered, config, encoder)
    
    # Step 6: Scale features
    print("\n[STEP 6] Scaling numerical features...")
    df_train_scaled, scaler = scale_features(df_train_encoded, config)
    df_production_scaled = scale_features(df_production_encoded, config, scaler)
    
    # Step 7: Save processed training data
    print("\n[STEP 7] Saving processed data...")
    version = config['versioning']['data_version']
    train_path = save_processed_data(df_train_scaled, f"{version}_train", config)
    
    # Save production data for monitoring
    prod_dir = config['data']['production_dir']
    os.makedirs(prod_dir, exist_ok=True)
    prod_path = os.path.join(prod_dir, f"{version}_production.csv")
    df_production_scaled.to_csv(prod_path, index=False)
    log_to_manifest(f"Saved production data: {prod_path}")

    # Create data summary
    print(f"\n\u001b[33m{'| DATA PREPARATION SUMMARY |':=^75}\u001b[0m")
    print(f"Training samples: {len(df_train_scaled)}")
    print(f"Production samples: {len(df_production_scaled)}")
    print(f"Number of features: {len(df_train_scaled.columns) - 1}")  # -1 for target
    print(f"Class distribution (training):")
    print(df_train_scaled[config["features"]["target"]].value_counts())
    print(f"\nData version: {version}")
    print(f"Training data: \u001b[32m{train_path}\u001b[0m")
    print(f"Production data: \u001b[32m{prod_path}\u001b[0m")
    print("=" * 70)
    
    # Save summary to manifest
    log_to_manifest(f"\n{'| DATA PREPARATION COMPLETE |':=^75}",time=False)
    log_to_manifest(f"Data version: {version}")
    log_to_manifest(f"Training samples: {len(df_train_scaled)}")
    log_to_manifest(f"Production samples: {len(df_production_scaled)}")

    git_hash = get_git_commit_hash()
    log_to_manifest(f"Git Commit Hash: {git_hash}")

if __name__ == "__main__":
    main()
