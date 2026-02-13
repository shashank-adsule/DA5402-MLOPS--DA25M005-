"""
Visualization Script
Create plots for understanding the data and model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def plot_class_distribution(config):
    """Plot class distribution in train vs production data"""
    
    # Load data
    train_df = pd.read_csv('data/processed/v1_train_processed.csv')
    prod_df = pd.read_csv('data/production/v1_production.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training data
    train_counts = train_df[config["features"]["target"]].value_counts()
    axes[0].bar(['No Failure', 'Failure'], 
                [train_counts[0], train_counts[1]],
                color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Training Data - Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xlabel('Class', fontsize=12)
    for i, v in enumerate([train_counts[0], train_counts[1]]):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Production data
    prod_counts = prod_df[config["features"]["target"]].value_counts()
    axes[1].bar(['No Failure', 'Failure'], 
                [prod_counts[0], prod_counts[1]],
                color=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Production Data - Class Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_xlabel('Class', fontsize=12)
    for i, v in enumerate([prod_counts[0], prod_counts[1]]):
        axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(config["data"]["plots_path"], exist_ok=True)        # to check if dir exists
    plt.savefig(f'{config["data"]["plots_path"]}plot_class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_class_distribution.png")
    plt.close()


def plot_feature_importance(config):
    """Plot feature importance from trained model"""
    # Load metadata
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Get top 10 features
    importance_data = metadata['feature_importance'][:10]
    
    features = [item['feature'] for item in importance_data]
    importances = [item['importance'] for item in importance_data]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    
    plt.barh(range(len(features)), importances, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add values on bars
    for i, v in enumerate(importances):
        plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{config["data"]["plots_path"]}plot_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_feature_importance.png")
    plt.close()


def plot_feature_drift(config):
    """Visualize feature drift between train and production"""
    # Load data
    train_df = pd.read_csv('data/processed/v1_train_processed.csv')
    prod_df = pd.read_csv('data/production/v1_production.csv')
    
    # Select key features to visualize
    features_to_plot = ['Tool wear [min]', 'Torque [Nm]', 
                        'Air temperature [K]', 'Power']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features_to_plot):
        # Plot distributions
        axes[idx].hist(train_df[feature], bins=30, alpha=0.6, 
                      label='Training', color='blue', density=True)
        axes[idx].hist(prod_df[feature], bins=30, alpha=0.6, 
                      label='Production', color='red', density=True)
        
        axes[idx].set_title(f'{feature} - Distribution Comparison', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Density', fontsize=10)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Feature Drift Analysis: Training vs Production', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{config["data"]["plots_path"]}plot_feature_drift.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_feature_drift.png")
    plt.close()


def plot_confusion_matrix(config):
    """Plot confusion matrix from monitoring results"""
    # Load monitoring log
    monitor_log = pd.read_csv('models/monitoring_log.csv')
    
    if len(monitor_log) == 0:
        print("No monitoring data available yet")
        return
    
    # Create confusion matrix visualization (using known values from monitor output)
    cm = np.array([[1868, 74],
                   [271, 787]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Production Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    
    # Add accuracy text
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(1, -0.3, f'Accuracy: {accuracy:.4f}', 
             ha='center', fontsize=12, fontweight='bold',
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{config["data"]["plots_path"]}plot_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_confusion_matrix.png")
    plt.close()


def plot_metrics_comparison(config):
    """Compare training vs production metrics"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    training_scores = [0.9993, 1.0000, 0.9966, 0.9983]
    production_scores = [0.8850, 0.9141, 0.7439, 0.8202]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, training_scores, width, 
                   label='Training', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, production_scores, width,
                   label='Production', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Training vs Production', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{config["data"]["plots_path"]}plot_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_metrics_comparison.png")
    plt.close()


def create_all_plots():
    """Create all visualization plots"""
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()
    
    config = load_config()
    os.makedirs(config["data"]["plots_path"],exist_ok=True)

    plot_class_distribution(config)
    plot_feature_importance(config)
    plot_feature_drift(config)
    plot_confusion_matrix(config)
    plot_metrics_comparison(config)
    
    print()
    print("=" * 70)
    print("All plots generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    create_all_plots()