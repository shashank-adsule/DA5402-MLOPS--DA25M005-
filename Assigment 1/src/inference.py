from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yaml
import json
import pickle
import os
from datetime import datetime
import csv

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and config
MODEL = None
CONFIG = None
METADATA = None
PREDICTION_LOG = None


def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_path):
    """Load trained model from disk"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_metadata(metadata_path):
    """Load model metadata"""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def log_deployment(config):
    """Log deployment event to deployment_log.csv"""
    log_path = config['logging']['deployment_log']
    
    # Check if file exists
    file_exists = os.path.isfile(log_path)
    
    # Prepare deployment record
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_version = config['versioning']['model_version']
    model_path = config['deployment']['model_path']
    
    # Write to CSV
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'timestamp', 'model_version', 'model_path', 
                'status', 'port', 'host'
            ])
        
        # Write deployment record
        writer.writerow([
            timestamp,
            model_version,
            model_path,
            'deployed',
            config['deployment']['api_port'],
            config['deployment']['api_host']
        ])
    
    print(f"✓ Deployment logged to {log_path}")


def log_prediction(input_data, prediction, probability, config):
    """Log each prediction for monitoring"""
    log_path = config['logging']['prediction_log']
    
    # Create directory if needed
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.isfile(log_path)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Write to CSV
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'timestamp', 'prediction', 'probability', 
                'model_version', 'input_features'
            ])
        
        # Write prediction record
        writer.writerow([
            timestamp,
            int(prediction),
            float(probability),
            config['versioning']['model_version'],
            json.dumps(input_data)
        ])


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'model_version': METADATA['model_version'] if METADATA else None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model information and metadata"""
    if METADATA is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_version': METADATA['model_version'],
        'data_version': METADATA['data_version'],
        'training_date': METADATA['training_date'],
        'algorithm': METADATA['algorithm'],
        'metrics': METADATA['metrics'],
        'git_commit': METADATA['git_commit']
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction on input data
    
    Expected input format:
    {
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.5,
        "Tool wear [min]": 50,
        "Type": "M"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert to DataFrame for prediction
        df = pd.DataFrame([data])
        
        # Feature engineering (must match training pipeline)
        if CONFIG['features'].get('create_power_feature', True):
            df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
        
        if CONFIG['features'].get('create_temp_diff', True):
            df['Temp_diff'] = (
                df['Process temperature [K]'] - df['Air temperature [K]']
            )
        
        if CONFIG['features'].get('create_interactions', True):
            df['Torque_ToolWear'] = df['Torque [Nm]'] * df['Tool wear [min]']
            df['Speed_ToolWear'] = (
                df['Rotational speed [rpm]'] * df['Tool wear [min]']
            )
        
        # Encode categorical (simple label encoding for Type)
        if 'Type' in df.columns:
            type_mapping = {'L': 0, 'M': 1, 'H': 2}
            df['Type_encoded'] = df['Type'].map(type_mapping)
            df = df.drop('Type', axis=1)
        
        # Ensure feature order matches training
        expected_features = METADATA['features']
        
        # Get only the features used in training
        # (excluding UDI, Product ID, Target, Failure Type)
        available_features = [f for f in expected_features if f in df.columns]
        df_features = df[available_features]
        
        # Make prediction
        prediction = MODEL.predict(df_features)[0]
        probability = MODEL.predict_proba(df_features)[0]
        
        # Get failure probability
        failure_probability = float(probability[1])
        
        # Apply threshold
        threshold = CONFIG['deployment']['prediction_threshold']
        prediction_binary = 1 if failure_probability >= threshold else 0
        
        # Log prediction
        log_prediction(data, prediction_binary, failure_probability, CONFIG)
        
        # Prepare response
        response = {
            'prediction': int(prediction_binary),
            'failure_probability': failure_probability,
            'prediction_class': 'Failure' if prediction_binary == 1 else 'No Failure',
            'threshold': threshold,
            'model_version': METADATA['model_version'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Make predictions on batch of data"""
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        predictions = []
        
        for sample in data['samples']:
            # Use single prediction endpoint logic
            df = pd.DataFrame([sample])
            
            # Feature engineering
            if CONFIG['features'].get('create_power_feature', True):
                df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
            
            if CONFIG['features'].get('create_temp_diff', True):
                df['Temp_diff'] = (
                    df['Process temperature [K]'] - df['Air temperature [K]']
                )
            
            if CONFIG['features'].get('create_interactions', True):
                df['Torque_ToolWear'] = df['Torque [Nm]'] * df['Tool wear [min]']
                df['Speed_ToolWear'] = (
                    df['Rotational speed [rpm]'] * df['Tool wear [min]']
                )
            
            # Encode categorical
            if 'Type' in df.columns:
                type_mapping = {'L': 0, 'M': 1, 'H': 2}
                df['Type_encoded'] = df['Type'].map(type_mapping)
                df = df.drop('Type', axis=1)
            
            # Get features
            expected_features = METADATA['features']
            available_features = [f for f in expected_features if f in df.columns]
            df_features = df[available_features]
            
            # Predict
            prediction = MODEL.predict(df_features)[0]
            probability = MODEL.predict_proba(df_features)[0]
            failure_probability = float(probability[1])
            
            threshold = CONFIG['deployment']['prediction_threshold']
            prediction_binary = 1 if failure_probability >= threshold else 0
            
            # Log prediction
            log_prediction(sample, prediction_binary, failure_probability, CONFIG)
            
            predictions.append({
                'prediction': int(prediction_binary),
                'failure_probability': failure_probability,
                'prediction_class': 'Failure' if prediction_binary == 1 else 'No Failure'
            })
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'model_version': METADATA['model_version'],
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def initialize_api():
    """Initialize API by loading model and config"""
    global MODEL, CONFIG, METADATA
    
    print(f"\u001b[33m{'INFERENCE API':=^75}\u001b[0m")
    
    # Load configuration
    CONFIG = load_config()
    print(f"\n✓ Loaded configuration")
    
    # Load model
    model_path = CONFIG['deployment']['model_path']
    
    if not os.path.exists(model_path):
        print(f"\n✗ ERROR: Model not found at {model_path}")
        print("Please run train.py first to create the model.")
        return False
    
    MODEL = load_model(model_path)
    print(f"✓ Loaded model from {model_path}")
    
    # Load metadata
    metadata_path = CONFIG['deployment']['metadata_path']
    METADATA = load_metadata(metadata_path)
    print(f"✓ Loaded metadata from {metadata_path}")
    
    # Log deployment
    log_deployment(CONFIG)
    
    print(f"\u001b[33m{'MODEL INFORMATION':=^75}\u001b[0m")
    
    print(f"Version: {METADATA['model_version']}")
    print(f"Algorithm: {METADATA['algorithm']}")
    print(f"Training Date: {METADATA['training_date']}")
    print(f"Data Version: {METADATA['data_version']}")
    print(f"Accuracy: {METADATA['metrics']['accuracy']:.4f}")
    print(f"Precision: {METADATA['metrics']['precision']:.4f}")
    print(f"Recall: {METADATA['metrics']['recall']:.4f}")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    # Initialize API
    if not initialize_api():
        exit(1)
    
    # Get host and port from config
    host = CONFIG['deployment']['api_host']
    port = CONFIG['deployment']['api_port']
    
    print(f"\n🚀 Starting API server on http://{host}:{port}")
    print(f"\nAvailable endpoints:")
    print(f"  - GET  /health          - Health check")
    print(f"  - GET  /model-info      - Model information")
    print(f"  - POST /predict         - Single prediction")
    print(f"  - POST /batch-predict   - Batch predictions")
    print("\nPress CTRL+C to stop the server")
    print("=" * 70 + "\n")
    
    # Run Flask app
    app.run(host=host, port=port, debug=False)