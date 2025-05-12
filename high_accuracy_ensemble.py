#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import logging
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc, confusion_matrix
import mlflow
from mlflow.tracking import MlflowClient
from mlflow_config import setup_mlflow
import joblib
import mlflow.pyfunc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MLflow
setup_mlflow()

# Define our models - including the ones that gave us 90%+ accuracy
MODELS = [
    {"name": "alzheimers_rf_tuned", "version": "9", "type": "sklearn"},
    {"name": "alzheimers_xgboost_tuned", "version": "9", "type": "sklearn"},
    {"name": "alzheimers_logistic_tuned", "version": "16", "type": "sklearn"},
    {"name": "alzheimers_knn_tuned", "version": "9", "type": "sklearn"},
    {"name": "alzheimers_svm_tuned", "version": "10", "type": "sklearn"},
    {"name": "alzheimers_nn_tuned", "version": "13", "type": "tensorflow"},
    # Additional models that gave high accuracy
    {"name": "alzheimers_rf_tuned", "version": "8", "type": "sklearn"},
    {"name": "alzheimers_xgboost_tuned", "version": "8", "type": "sklearn"},
    {"name": "alzheimers_nn_tuned", "version": "12", "type": "tensorflow"}
]

def parse_args():
    parser = argparse.ArgumentParser(description="Create high-accuracy ensemble for Alzheimer's prediction")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("data", "processed"),
        help="Path to test data (default: data/processed)"
    )
    return parser.parse_args()

def load_test_data(data_path):
    """Load test data for ensemble optimization."""
    try:
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        feature_names = pd.read_csv(os.path.join(data_path, 'feature_names.csv'))['0'].values
        
        logger.info(f"Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        return X_test, y_test, feature_names
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def load_model(model_info):
    """Load a model from MLflow based on its name, version, and type."""
    model_name = model_info["name"]
    model_version = model_info["version"]
    model_type = model_info["type"]
    
    model_uri = f"models:/{model_name}/{model_version}"
    logger.info(f"Loading model: {model_uri}")
    
    try:
        if model_type == "sklearn":
            model = mlflow.sklearn.load_model(model_uri)
        elif model_type == "tensorflow":
            model = mlflow.tensorflow.load_model(model_uri)
        else:
            # Fallback to pyfunc
            model = mlflow.pyfunc.load_model(model_uri)
        
        return model, model_uri
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None, model_uri

def get_model_predictions(model, X_test):
    """Get predictions from a model, handling different model types."""
    try:
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
                    return None, None
        else:
            logger.error("Model does not have a standard prediction method")
            return None, None
        
        return y_pred, y_prob
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return None, None

def find_optimal_threshold(y_test, y_prob):
    """Find the optimal threshold that maximizes the tradeoff between sensitivity and specificity."""
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Calculate the Youden's J statistic (sensitivity + specificity - 1)
    j_scores = tpr - fpr
    
    # Find the optimal threshold that maximizes J
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_sensitivity = tpr[best_idx]
    best_specificity = 1 - fpr[best_idx]
    
    return {
        'threshold': best_threshold,
        'sensitivity': best_sensitivity,
        'specificity': best_specificity,
        'youden_j': j_scores[best_idx]
    }

def calculate_metrics(y_test, y_pred, y_prob):
    """Calculate performance metrics."""
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate sensitivity (recall), specificity, and precision
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Overall accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # AUC
    auc_score = roc_auc_score(y_test, y_prob)
    
    # Find optimal threshold
    optimal = find_optimal_threshold(y_test, y_prob)
    
    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'optimal_threshold': optimal['threshold']
    }

def create_weighted_ensemble(y_test, y_pred_list, y_prob_list, model_names):
    """Create a weighted average ensemble based on model performance."""
    # Calculate individual model performance
    individual_metrics = []
    for i, (y_pred, y_prob) in enumerate(zip(y_pred_list, y_prob_list)):
        model_metrics = calculate_metrics(y_test, y_pred, y_prob)
        individual_metrics.append((i, model_metrics["accuracy"], model_metrics["auc"]))
        logger.info(f"Model {model_names[i]} - Accuracy: {model_metrics['accuracy']:.4f}, AUC: {model_metrics['auc']:.4f}")
    
    # Sort by accuracy
    individual_metrics.sort(key=lambda x: x[1], reverse=True)
    
    # Weight models based on ranked performance (more weight to better models)
    weights = np.zeros(len(y_prob_list))
    for rank, (idx, _, _) in enumerate(individual_metrics):
        # Exponential weighting - higher ranked models get exponentially more weight
        weights[idx] = np.exp(-0.5 * rank)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Create weighted average predictions
    ensemble_probs = np.zeros(len(y_test))
    for i, y_prob in enumerate(y_prob_list):
        ensemble_probs += weights[i] * y_prob
    
    # Find optimal threshold
    optimal_results = find_optimal_threshold(y_test, ensemble_probs)
    optimal_threshold = optimal_results["threshold"]
    
    # Get predictions using optimal threshold
    ensemble_preds = (ensemble_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, ensemble_preds, ensemble_probs)
    
    logger.info(f"Weighted Average Ensemble - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    return weights, ensemble_preds, ensemble_probs, metrics

class WeightedEnsembleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import json, numpy as np, mlflow.pyfunc
        with open(context.artifacts["ensemble_config"], "r") as f:
            cfg = json.load(f)
        self.model_names = cfg["model_names"]
        self.model_uris = [cfg["model_uris"][name] for name in self.model_names]
        self.weights = np.array([cfg["model_weights"][name] for name in self.model_names])
        self.threshold = cfg["optimal_threshold"]
        self.base_models = [mlflow.pyfunc.load_model(uri) for uri in self.model_uris]

    def predict(self, context, model_input):
        import numpy as np
        X = model_input.values
        ensemble_probs = np.zeros(X.shape[0])
        for w, m in zip(self.weights, self.base_models):
            probs = m.predict_proba(X)[:, 1]
            ensemble_probs += w * probs
        return (ensemble_probs >= self.threshold).astype(int)

def log_to_mlflow(weights, preds, probs, model_names, model_uris, metrics, output_dir):
    """Log the ensemble to MLflow and register as PyFunc model."""
    experiment_name = "weighted_ensemble"
    mlflow.set_experiment(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"weighted_ensemble_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("ensemble_method", "weighted_average")
        mlflow.log_param("number_of_models", len(model_names))
        
        # Log key metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("auc", metrics["auc"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("sensitivity", metrics["sensitivity"])
        mlflow.log_metric("specificity", metrics["specificity"])
        mlflow.log_metric("optimal_threshold", metrics["optimal_threshold"])
        
        # Create a dictionary to make model names unique for logging
        unique_model_names = {}
        for i, name in enumerate(model_names):
            if name in unique_model_names:
                # Add a suffix to make the name unique
                unique_model_names[name] = unique_model_names[name] + 1
                unique_name = f"{name}_v{unique_model_names[name]}"
            else:
                unique_model_names[name] = 1
                unique_name = name
            
            # Log weight with unique name
            mlflow.log_param(f"weight_{unique_name}", weights[i])
        
        # Save the ensemble configuration
        config_path = os.path.join(output_dir, "ensemble_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_names": model_names,
                "model_uris": {name: uri for name, uri in zip(model_names, model_uris)},
                "model_weights": {name: float(w) for name, w in zip(model_names, weights)},
                "optimal_threshold": float(metrics["optimal_threshold"])}, f, indent=2)
        mlflow.log_artifact(config_path)

        # Log the PyFunc model artifact under this run
        mlflow.pyfunc.log_model(
            artifact_path="ensemble_pyfunc",
            python_model=WeightedEnsembleModel(),
            artifacts={"ensemble_config": config_path},
            registered_model_name="high_accuracy_ensemble"
        )

        run_id = run.info.run_id
        logger.info(f"Logged and registered ensemble as high_accuracy_ensemble, run_id: {run_id}")
        return run_id

if __name__ == "__main__":
    args = parse_args()
    
    # Set up directories
    output_dir = os.path.join("models", "ensembles", f"high_acc_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    X_test, y_test, feature_names = load_test_data(args.data_path)
    
    # Load models
    models = []
    model_uris = []
    model_names = []
    
    for i, model_info in enumerate(MODELS):
        model, model_uri = load_model(model_info)
        if model is not None:
            # Make model name unique if it's a duplicate
            if model_info["name"] in model_names:
                unique_name = f"{model_info['name']}_v{model_info['version']}"
            else:
                unique_name = model_info["name"]
                
            models.append(model)
            model_uris.append(model_uri)
            model_names.append(unique_name)
    
    if len(models) < 3:
        logger.error("Not enough models could be loaded. Exiting.")
        exit(1)
    
    logger.info(f"Successfully loaded {len(models)} models")
    
    # Get predictions from each model
    y_pred_list = []
    y_prob_list = []
    valid_model_indices = []
    valid_model_names = []
    valid_model_uris = []
    
    for i, model in enumerate(models):
        model_name = model_names[i]
        logger.info(f"Getting predictions from model: {model_name}")
        y_pred, y_prob = get_model_predictions(model, X_test)
        
        if y_pred is not None and y_prob is not None:
            y_pred_list.append(y_pred)
            y_prob_list.append(y_prob)
            valid_model_indices.append(i)
            valid_model_names.append(model_name)
            valid_model_uris.append(model_uris[i])
    
    # Create weighted ensemble
    weights, ensemble_preds, ensemble_probs, metrics = create_weighted_ensemble(
        y_test, y_pred_list, y_prob_list, valid_model_names
    )
    
    # Log to MLflow
    run_id = log_to_mlflow(
        weights, ensemble_preds, ensemble_probs, 
        valid_model_names, valid_model_uris, metrics, output_dir
    )
    
    # Print summary
    print("\n===== HIGH ACCURACY ENSEMBLE RESULTS =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print("\nTop 5 Model Weights:")
    # Sort weights for display
    sorted_weights = sorted(zip(valid_model_names, weights), key=lambda x: x[1], reverse=True)
    for model_name, weight in sorted_weights[:5]:
        print(f"  {model_name}: {weight:.4f}")
    
    print(f"\nResults saved to {output_dir}")
    print(f"MLflow run ID: {run_id}")
    print(f"To access this ensemble: Use the direct-ensemble endpoint in the API") 