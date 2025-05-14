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
from mlflow_config import setup_mlflow

setup_mlflow()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models to be monitored with their versions
MODELS_TO_MONITOR = [
    {"name": "alzheimers_rf_tuned", "version": "9"},
    {"name": "alzheimers_xgboost_tuned", "version": "9"},
    {"name": "alzheimers_logistic_tuned", "version": "16"},
    {"name": "alzheimers_knn_tuned", "version": "9"},
    {"name": "alzheimers_svm_tuned", "version": "10"},
    {"name": "alzheimers_nn_tuned", "version": "13"}
]

MONITORING_DATA_PATH = os.path.join("data", "monitoring")

def parse_args():
    """Parse command line arguments for model monitoring"""
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
        "--all",
        action="store_true",
        help="Monitor all models"
    )
    return parser.parse_args()

def load_model_from_registry(model_name, model_version=None):
    """Load model from MLflow registry with specified name and version"""
    client = MlflowClient()
    
    try:
        if model_version is not None:
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                logger.error(f"No Production version found for model {model_name}")
                return None, None
                
            model_version = versions[0].version
            model_uri = f"models:/{model_name}/Production"
        
        logger.info(f"Loading model: {model_uri}")
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model as sklearn flavor")
        except Exception as sklearn_error:
            try:
                model = mlflow.tensorflow.load_model(model_uri)
                logger.info(f"Loaded model as tensorflow flavor")
            except Exception as tf_error:
                try:
                    model = mlflow.pyfunc.load_model(model_uri)
                    logger.info(f"Loaded model as pyfunc flavor")
                except Exception as pyfunc_error:
                    logger.error(f"Failed to load model with any flavor")
                    return None, None
        
        return model, model_uri
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def load_latest_test_data():
    """Load the latest monitoring test data"""
    data_dir = MONITORING_DATA_PATH
    logger.info(f"Using monitoring data from {data_dir}")
    
    if os.path.exists(os.path.join(data_dir, "data_history.json")):
        try:
            with open(os.path.join(data_dir, "data_history.json"), 'r') as f:
                history = json.load(f)
            
            if history:
                latest_batch = history[-1]["timestamp"]
                batch_dir = os.path.join(data_dir, f"batch_{latest_batch}")
                
                if os.path.exists(batch_dir):
                    data_dir = batch_dir
                    logger.info(f"Using latest batch: {latest_batch}")
        except Exception as e:
            logger.warning(f"Could not determine latest batch: {e}")
    
    try:
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        feature_names = pd.read_csv(os.path.join(data_dir, 'feature_names.csv'))['0'].values
        
        logger.info(f"Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        return X_test, y_test, feature_names
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def evaluate_model_performance(model, X_test, y_test):
    """Evaluate model performance on test data and return metrics"""
    try:
        if hasattr(model, "predict_proba"):
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "predict"):
            predictions = model.predict(X_test)
            
            if isinstance(predictions, np.ndarray):
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    y_prob = predictions[:, 1]
                    y_pred = np.argmax(predictions, axis=1)
                else:
                    y_prob = predictions.flatten()
                    y_pred = (y_prob > 0.5).astype(int)
            else:
                try:
                    y_prob = np.array(predictions).flatten()
                    y_pred = (y_prob > 0.5).astype(int)
                except:
                    logger.error("Could not convert model predictions")
                    y_prob = np.zeros(len(y_test))
                    y_pred = np.zeros(len(y_test))
        else:
            logger.error("Model does not have a standard prediction method")
            return {
                "accuracy": 0.0,
                "auc": 0.5,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, np.zeros(len(y_test)), np.zeros(len(y_test))
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Model performance: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['auc']:.4f}")
        return metrics, y_pred, y_prob
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {
            "accuracy": 0.0,
            "auc": 0.5,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, np.zeros(len(y_test)), np.zeros(len(y_test))

def detect_performance_drift(metrics, threshold=0.05):
    """Detect if model performance has drifted beyond the threshold"""
    history_file = os.path.join("monitoring", "performance_history.json")
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        if len(history) > 0:
            avg_accuracy = np.mean([h["accuracy"] for h in history])
            avg_auc = np.mean([h["auc"] for h in history])
            
            accuracy_drift = abs(metrics["accuracy"] - avg_accuracy)
            auc_drift = abs(metrics["auc"] - avg_auc)
            
            if accuracy_drift > threshold or auc_drift > threshold:
                logger.warning(f"Performance drift detected! Accuracy drift: {accuracy_drift:.4f}, AUC drift: {auc_drift:.4f}")
                return True
    
    return False

def save_performance_metrics(metrics):
    """Save model performance metrics to history file"""
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
    """Generate monitoring plots for model performance visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
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
    
    # ROC Curve
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
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
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
    """Generate random batches of test data for monitoring"""
    if batch_size is None:
        batch_size = len(X_test) // n_batches
    
    batches = []
    indices = np.arange(len(X_test))
    
    for i in range(n_batches):
        batch_indices = np.random.choice(indices, size=batch_size, replace=False)
        X_batch = X_test[batch_indices]
        y_batch = y_test[batch_indices]
        batches.append((X_batch, y_batch))
        
    return batches

def setup_monitoring_log():
    """Setup logging for drift alerts"""
    log_dir = "monitoring/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "drift_alerts.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.WARNING)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return log_file

def plot_batch_performance(model_name, batch_metrics, output_dir):
    """Plot performance metrics across batches to visualize drift"""
    plt.figure(figsize=(12, 6))
    
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
    """Detect if performance has drifted across recent batches"""
    auc_scores = [m['auc'] for m in batch_metrics]
    accuracy_scores = [m['accuracy'] for m in batch_metrics]
    
    mean_auc = np.mean(auc_scores)
    mean_accuracy = np.mean(accuracy_scores)
    
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

def monitor_single_model(model_name, model_version):
    """Monitor a single model's performance and detect drift"""
    model, model_uri = load_model_from_registry(model_name, model_version)
    if model is None:
        logger.error(f"Failed to load model {model_name}")
        return False
    
    try:
        X_test, y_test, feature_names = load_latest_test_data()
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return False
    
    batches = generate_data_batches(X_test, y_test, n_batches=10)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = setup_monitoring_log()
    
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
    
    run_name = f"model_monitoring_{model_name}"
    if model_version:
        run_name += f"_v{model_version}"
    run_name += f"_{timestamp}"
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("monitoring_time", timestamp)
        
        batch_metrics = []
        
        for i, (X_batch, y_batch) in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/10")
            
            metrics, y_pred, y_prob = evaluate_model_performance(model, X_batch, y_batch)
            batch_metrics.append(metrics)
            
            for key, value in metrics.items():
                if key != 'timestamp':
                    mlflow.log_metric(f"batch_{i}_{key}", value)
        
        output_dir = os.path.join("monitoring", f"plots_{model_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        plot_batch_performance(model_name, batch_metrics, output_dir)
        
        drift_detected, drift_message = detect_batch_drift(batch_metrics)
        
        if drift_detected:
            logger.warning(f"Model: {model_name} - {drift_message}")
        
        mlflow.log_artifacts(output_dir)
        
        logger.info(f"Model monitoring completed for {model_name}. MLflow run ID: {run.info.run_id}")
        return True

def monitor_all_models():
    """Monitor all models defined in MODELS_TO_MONITOR"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        "timestamp": timestamp,
        "results": {}
    }
    
    logger.info(f"Monitoring all {len(MODELS_TO_MONITOR)} models")
    summary["models_monitored"] = len(MODELS_TO_MONITOR)
    
    for model_info in MODELS_TO_MONITOR:
        model_name = model_info["name"]
        model_version = model_info["version"]
        start_time = datetime.now()
        
        success = monitor_single_model(
            model_name,
            model_version
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary["results"][model_name] = {
            "success": success,
            "version": model_version,
            "duration_seconds": duration
        }
    
    save_monitoring_summary(summary)
    
    logger.info(f"Monitoring completed for {len(MODELS_TO_MONITOR)} models")
    successes = sum(1 for model, result in summary["results"].items() if result["success"])
    logger.info(f"Successful: {successes}/{len(MODELS_TO_MONITOR)}")
    
    return summary

def save_monitoring_summary(results, output_path="monitoring/summary"):
    """Save monitoring results summary to JSON file"""
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_path, f"monitoring_summary_{timestamp}.json")
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved monitoring summary to {summary_file}")

def main():
    args = parse_args()
    
    if args.all or args.model_name is None:
        monitor_all_models()
    else:
        monitor_single_model(
            args.model_name,
            args.model_version
        )

if __name__ == "__main__":
    main() 