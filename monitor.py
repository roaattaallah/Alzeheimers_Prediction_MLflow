#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import subprocess

# Import MLflow configuration
from mlflow_config import setup_mlflow

# Set up MLflow tracking
setup_mlflow()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define models to monitor
MODELS_TO_MONITOR = [
    {"name": "alzheimers_rf_tuned", "version": "9"},
    {"name": "alzheimers_xgboost_tuned", "version": "9"},
    {"name": "alzheimers_logistic_tuned", "version": "16"},
    {"name": "alzheimers_knn_tuned", "version": "9"},
    {"name": "alzheimers_svm_tuned", "version": "10"},
    {"name": "alzheimers_nn_tuned", "version": "13"}
]

# Fixed monitoring data path
MONITORING_DATA_PATH = os.path.join("data", "monitoring")

def parse_args():
    parser = argparse.ArgumentParser(description="Model monitoring script")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model in MLflow Model Registry to monitor. If not provided, all models will be monitored."
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default=None,
        help="Version of the model to monitor. If not provided, the Production version will be used."
    )
    parser.add_argument(
        "--simulate_drift",
        action='store_true',
        help="Simulate data drift by adding noise to test data"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Monitor all models"
    )
    return parser.parse_args()

def load_model_from_registry(model_name, model_version=None):
    """
    Load a model from the MLflow Model Registry.
    
    Parameters:
    -----------
    model_name : str
        Name of the model in MLflow Model Registry.
    model_version : int, optional
        Version of the model to load. If not provided, the Production version will be used.
        
    Returns:
    --------
    model : object
        The loaded model.
    model_uri : str
        URI of the loaded model.
    """
    client = MlflowClient()
    
    try:
        if model_version is not None:
            # Get specific version
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            # Get Production version
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                logger.error(f"No Production version found for model {model_name}")
                return None, None
                
            model_version = versions[0].version
            model_uri = f"models:/{model_name}/Production"
        
        logger.info(f"Loading model: {model_uri}")
        
        # Try to determine the model type and load accordingly
        try:
            # First try sklearn flavor (most common)
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model as sklearn flavor")
        except Exception as sklearn_error:
            try:
                # Try tensorflow/keras flavor
                model = mlflow.tensorflow.load_model(model_uri)
                logger.info(f"Loaded model as tensorflow flavor")
            except Exception as tf_error:
                try:
                    # Try pyfunc as a fallback (works with most model types)
                    model = mlflow.pyfunc.load_model(model_uri)
                    logger.info(f"Loaded model as pyfunc flavor")
                except Exception as pyfunc_error:
                    logger.error(f"Failed to load model with any flavor: {sklearn_error}, {tf_error}, {pyfunc_error}")
                    return None, None
        
        return model, model_uri
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def load_latest_test_data():
    """
    Load the latest batch of test data for model monitoring.
    
    Returns:
    --------
    X_test, y_test, feature_names : numpy arrays
        Test data features, labels, and feature names.
    """
    data_dir = MONITORING_DATA_PATH
    logger.info(f"Using monitoring data from {data_dir}")
    
    # Find the latest batch if data_history.json exists
    if os.path.exists(os.path.join(data_dir, "data_history.json")):
        try:
            with open(os.path.join(data_dir, "data_history.json"), 'r') as f:
                history = json.load(f)
            
            if history:
                # Get the latest batch timestamp
                latest_batch = history[-1]["timestamp"]
                batch_dir = os.path.join(data_dir, f"batch_{latest_batch}")
                
                if os.path.exists(batch_dir):
                    data_dir = batch_dir
                    logger.info(f"Using latest batch: {latest_batch}")
        except Exception as e:
            logger.warning(f"Could not determine latest batch: {e}. Using base monitoring directory.")
    
    # Load the data from the determined directory
    try:
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        feature_names = pd.read_csv(os.path.join(data_dir, 'feature_names.csv'))['0'].values
        
        logger.info(f"Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        return X_test, y_test, feature_names
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def add_synthetic_drift(X_test, magnitude=1.0):
    """
    Simulate data drift by adding noise to features.
    
    Parameters:
    -----------
    X_test : numpy array
        Test data features.
    magnitude : float
        Magnitude of the drift to simulate.
        
    Returns:
    --------
    X_test_drift : numpy array
        Test data with simulated drift.
    """
    # Get standard deviation of each feature
    feature_std = np.std(X_test, axis=0)
    
    # Add random noise proportional to feature std dev
    noise = np.random.normal(0, feature_std * magnitude, X_test.shape)
    X_test_drift = X_test + noise
    
    logger.info(f"Added synthetic drift with magnitude {magnitude}")
    return X_test_drift

def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluate model performance metrics.
    
    Parameters:
    -----------
    model : object
        The model to evaluate.
    X_test, y_test : numpy arrays
        Test data features and labels.
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics.
    y_pred, y_prob : numpy arrays
        Predicted labels and probabilities.
    """
    try:
        # Handle different model types for predictions
        if hasattr(model, "predict_proba"):
            # Standard sklearn models
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "predict"):
            # For tensorflow/keras models or pyfunc models
            predictions = model.predict(X_test)
            
            # Handle different prediction formats
            if isinstance(predictions, np.ndarray):
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    # Multi-class predictions
                    y_prob = predictions[:, 1]
                    y_pred = np.argmax(predictions, axis=1)
                else:
                    # Binary predictions as probabilities
                    y_prob = predictions.flatten()
                    y_pred = (y_prob > 0.5).astype(int)
            else:
                # Handle other prediction formats
                try:
                    y_prob = np.array(predictions).flatten()
                    y_pred = (y_prob > 0.5).astype(int)
                except:
                    logger.error("Could not convert model predictions to probabilities")
                    y_prob = np.zeros(len(y_test))
                    y_pred = np.zeros(len(y_test))
        else:
            logger.error("Model does not have a standard prediction method")
            return {
                "accuracy": 0.0,
                "auc": 0.5,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, np.zeros(len(y_test)), np.zeros(len(y_test))
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Model performance: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['auc']:.4f}")
        return metrics, y_pred, y_prob
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        # Return default metrics
        return {
            "accuracy": 0.0,
            "auc": 0.5,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, np.zeros(len(y_test)), np.zeros(len(y_test))

def detect_performance_drift(metrics, threshold=0.05):
    """
    Detect if there's a significant drift in model performance.
    
    Parameters:
    -----------
    metrics : dict
        Current performance metrics.
    threshold : float
        Threshold for detecting drift.
        
    Returns:
    --------
    drift_detected : bool
        True if drift is detected, False otherwise.
    """
    # Load historical metrics if available
    history_file = os.path.join("monitoring", "performance_history.json")
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        if len(history) > 0:
            # Calculate average historical performance
            avg_accuracy = np.mean([h["accuracy"] for h in history])
            avg_auc = np.mean([h["auc"] for h in history])
            
            # Check for significant drift
            accuracy_drift = abs(metrics["accuracy"] - avg_accuracy)
            auc_drift = abs(metrics["auc"] - avg_auc)
            
            if accuracy_drift > threshold or auc_drift > threshold:
                logger.warning(f"Performance drift detected! Accuracy drift: {accuracy_drift:.4f}, AUC drift: {auc_drift:.4f}")
                return True
    
    return False

def save_performance_metrics(metrics):
    """
    Save performance metrics to a history file.
    
    Parameters:
    -----------
    metrics : dict
        Performance metrics to save.
    """
    os.makedirs("monitoring", exist_ok=True)
    history_file = os.path.join("monitoring", "performance_history.json")
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(metrics)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Performance metrics saved to {history_file}")

def generate_monitoring_plots(model, X_test, y_test, y_pred, y_prob, feature_names, output_dir):
    """
    Generate monitoring plots.
    
    Parameters:
    -----------
    model : object
        The model being monitored.
    X_test, y_test : numpy arrays
        Test data features and labels.
    y_pred, y_prob : numpy arrays
        Predicted labels and probabilities.
    feature_names : numpy array
        Names of the features.
    output_dir : str
        Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Alzheimer\'s', 'Alzheimer\'s'],
                yticklabels=['No Alzheimer\'s', 'Alzheimer\'s'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # 4. Feature Distribution (top 5 most important features)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = indices[:5]
        
        plt.figure(figsize=(12, 10))
        for i, idx in enumerate(top_features):
            plt.subplot(3, 2, i+1)
            for label in [0, 1]:
                sns.kdeplot(X_test[y_test == label, idx], label=f'Class {label}')
            plt.title(f'Feature: {feature_names[idx]}')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
        plt.close()
    
    logger.info(f"Monitoring plots saved to {output_dir}")

def generate_data_batches(X_test, y_test, n_batches=10, batch_size=None):
    """
    Generate batches from the monitoring dataset.
    
    Parameters:
    -----------
    X_test : numpy array
        Test data features
    y_test : numpy array
        Test data labels
    n_batches : int
        Number of batches to generate
    batch_size : int, optional
        Size of each batch. If None, will be calculated based on data size
        
    Returns:
    --------
    list of tuples
        List of (X_batch, y_batch) for each batch
    """
    if batch_size is None:
        batch_size = len(X_test) // n_batches
    
    batches = []
    indices = np.arange(len(X_test))
    
    for i in range(n_batches):
        # Randomly sample indices for this batch
        batch_indices = np.random.choice(indices, size=batch_size, replace=False)
        X_batch = X_test[batch_indices]
        y_batch = y_test[batch_indices]
        batches.append((X_batch, y_batch))
        
    return batches

def setup_monitoring_log():
    """Set up a dedicated log file for drift alerts."""
    log_dir = "monitoring/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a file handler
    log_file = os.path.join(log_dir, "drift_alerts.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.WARNING)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    return log_file

def plot_batch_performance(model_name, batch_metrics, output_dir):
    """
    Create visualizations for batch performance.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    batch_metrics : list
        List of metrics for each batch
    output_dir : str
        Directory to save plots
    """
    plt.figure(figsize=(12, 6))
    
    # Plot ROC AUC scores across batches
    batch_numbers = range(1, len(batch_metrics) + 1)
    auc_scores = [m['auc'] for m in batch_metrics]
    accuracy_scores = [m['accuracy'] for m in batch_metrics]
    
    plt.subplot(1, 2, 1)
    plt.plot(batch_numbers, auc_scores, marker='o', label='ROC AUC')
    plt.axhline(y=np.mean(auc_scores), color='r', linestyle='--', label='Mean AUC')
    plt.fill_between(batch_numbers, 
                     np.mean(auc_scores) - 0.05, 
                     np.mean(auc_scores) + 0.05, 
                     alpha=0.2, color='r', label='Drift Threshold')
    plt.xlabel('Batch Number')
    plt.ylabel('ROC AUC Score')
    plt.title(f'{model_name} - ROC AUC Performance')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(batch_numbers, accuracy_scores, marker='o', label='Accuracy')
    plt.axhline(y=np.mean(accuracy_scores), color='r', linestyle='--', label='Mean Accuracy')
    plt.fill_between(batch_numbers, 
                     np.mean(accuracy_scores) - 0.05, 
                     np.mean(accuracy_scores) + 0.05, 
                     alpha=0.2, color='r', label='Drift Threshold')
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy Score')
    plt.title(f'{model_name} - Accuracy Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_batch_performance.png'))
    plt.close()

def detect_batch_drift(batch_metrics, threshold=0.05):
    """
    Detect drift across batches.
    
    Parameters:
    -----------
    batch_metrics : list
        List of metrics for each batch
    threshold : float
        Threshold for drift detection
        
    Returns:
    --------
    bool, str
        Whether drift was detected and description of the drift
    """
    auc_scores = [m['auc'] for m in batch_metrics]
    accuracy_scores = [m['accuracy'] for m in batch_metrics]
    
    mean_auc = np.mean(auc_scores)
    mean_accuracy = np.mean(accuracy_scores)
    
    # Check for drift in recent batches (last 3)
    recent_auc_drift = any(abs(auc - mean_auc) > threshold for auc in auc_scores[-3:])
    recent_acc_drift = any(abs(acc - mean_accuracy) > threshold for acc in accuracy_scores[-3:])
    
    if recent_auc_drift or recent_acc_drift:
        drift_msg = f"Performance drift detected! "
        if recent_auc_drift:
            drift_msg += f"AUC deviation exceeds threshold of {threshold}. "
        if recent_acc_drift:
            drift_msg += f"Accuracy deviation exceeds threshold of {threshold}."
        return True, drift_msg
    
    return False, "No significant drift detected"

def monitor_single_model(model_name, model_version, simulate_drift):
    """Monitor a single model across multiple batches and return the results."""
    # Load model from registry
    model, model_uri = load_model_from_registry(model_name, model_version)
    if model is None:
        logger.error(f"Failed to load model {model_name}")
        return False
    
    # Load test data
    try:
        X_test, y_test, feature_names = load_latest_test_data()
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return False
    
    # Generate batches
    batches = generate_data_batches(X_test, y_test, n_batches=10)
    
    # Create monitoring timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup monitoring log
    log_file = setup_monitoring_log()
    
    # Set MLflow experiment
    experiment_name = "alzheimers_model_monitoring"
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        elif experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
            
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        logger.info("Using default experiment")
    
    # Create a unique run name
    run_name = f"model_monitoring_{model_name}"
    if model_version:
        run_name += f"_v{model_version}"
    run_name += f"_{timestamp}"
    
    # Start MLflow run for monitoring
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("monitoring_time", timestamp)
        
        # Store metrics for each batch
        batch_metrics = []
        
        # Process each batch
        for i, (X_batch, y_batch) in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/10")
            
            # Apply drift if simulating
            if simulate_drift:
                X_batch = add_synthetic_drift(X_batch, magnitude=0.1 * i)  # Increasing drift
            
            # Evaluate the model on this batch
            metrics, y_pred, y_prob = evaluate_model_performance(model, X_batch, y_batch)
            batch_metrics.append(metrics)
            
            # Log batch metrics to MLflow
            for key, value in metrics.items():
                if key != 'timestamp':
                    mlflow.log_metric(f"batch_{i}_{key}", value)
        
        # Generate performance visualization
        output_dir = os.path.join("monitoring", f"plots_{model_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        plot_batch_performance(model_name, batch_metrics, output_dir)
        
        # Detect drift across batches
        drift_detected, drift_message = detect_batch_drift(batch_metrics)
        
        if drift_detected:
            logger.warning(f"Model: {model_name} - {drift_message}")
        
        # Log artifacts
        mlflow.log_artifacts(output_dir)
        
        logger.info(f"Model monitoring completed for {model_name}. MLflow run ID: {run.info.run_id}")
        return True

def monitor_all_models(simulate_drift):
    """Monitor all models."""
    # Create timestamp for this monitoring run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a summary dictionary to track results
    summary = {
        "timestamp": timestamp,
        "simulate_drift": simulate_drift,
        "results": {}
    }
    
    logger.info(f"Monitoring all {len(MODELS_TO_MONITOR)} models")
    summary["models_monitored"] = len(MODELS_TO_MONITOR)
    
    # Monitor each model
    for model_info in MODELS_TO_MONITOR:
        model_name = model_info["name"]
        model_version = model_info["version"]
        start_time = datetime.now()
        
        success = monitor_single_model(
            model_name,
            model_version,
            simulate_drift
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Store result in summary
        summary["results"][model_name] = {
            "success": success,
            "version": model_version,
            "duration_seconds": duration
        }
    
    # Save monitoring summary
    save_monitoring_summary(summary)
    
    # Log completion
    logger.info(f"Monitoring completed for {len(MODELS_TO_MONITOR)} models")
    successes = sum(1 for model, result in summary["results"].items() if result["success"])
    logger.info(f"Successful: {successes}/{len(MODELS_TO_MONITOR)}")
    
    return summary

def save_monitoring_summary(results, output_path="monitoring/summary"):
    """Save a summary of the monitoring run."""
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_path, f"monitoring_summary_{timestamp}.json")
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved monitoring summary to {summary_file}")

def main():
    args = parse_args()
    
    # Check if we're monitoring a single model or all models
    if args.all or args.model_name is None:
        # Monitor all models
        monitor_all_models(args.simulate_drift)
    else:
        # Monitor a single model
        monitor_single_model(
            args.model_name,
            args.model_version,
            args.simulate_drift
        )

if __name__ == "__main__":
    main() 