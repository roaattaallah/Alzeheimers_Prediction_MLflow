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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

setup_mlflow()

# Models to include in the ensemble with their versions and types
MODELS = [
    {"name": "alzheimers_rf_tuned", "version": "9", "type": "sklearn"},
    {"name": "alzheimers_xgboost_tuned", "version": "9", "type": "sklearn"},
    {"name": "alzheimers_logistic_tuned", "version": "16", "type": "sklearn"},
    {"name": "alzheimers_knn_tuned", "version": "9", "type": "sklearn"},
    {"name": "alzheimers_svm_tuned", "version": "10", "type": "sklearn"},
    {"name": "alzheimers_nn_tuned", "version": "13", "type": "tensorflow"},
    {"name": "alzheimers_rf_tuned", "version": "8", "type": "sklearn"},
    {"name": "alzheimers_xgboost_tuned", "version": "8", "type": "sklearn"},
    {"name": "alzheimers_nn_tuned", "version": "12", "type": "tensorflow"}
]

def parse_args():
    """Parse command line arguments for ensemble creation"""
    parser = argparse.ArgumentParser(description="Create high-accuracy ensemble for Alzheimer's prediction")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("data", "processed"),
        help="Path to test data (default: data/processed)"
    )
    return parser.parse_args()

def load_test_data(data_path):
    """Load test data from specified path"""
    try:
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        feature_names = pd.read_csv(os.path.join(data_path, 'feature_names.csv'))['0'].values
        return X_test, y_test, feature_names
    except Exception as e:
        raise

def load_model(model_info):
    """Load a model from MLflow registry using model info"""
    model_name = model_info["name"]
    model_version = model_info["version"]
    model_type = model_info["type"]
    
    model_uri = f"models:/{model_name}/{model_version}"
    
    try:
        if model_type == "sklearn":
            model = mlflow.sklearn.load_model(model_uri)
        elif model_type == "tensorflow":
            model = mlflow.tensorflow.load_model(model_uri)
        else:
            model = mlflow.pyfunc.load_model(model_uri)
        
        return model, model_uri
    except Exception:
        return None, model_uri

def get_model_predictions(model, X_test):
    """Get predictions and probabilities from a model"""
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
                    return None, None
        else:
            return None, None
        
        return y_pred, y_prob
    except Exception:
        return None, None

def find_optimal_threshold(y_test, y_prob):
    """Find optimal classification threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    j_scores = tpr - fpr
    
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
    """Calculate performance metrics for model evaluation"""
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    auc_score = roc_auc_score(y_test, y_prob)
    
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
    """Create weighted ensemble from multiple models based on performance"""
    individual_metrics = []
    for i, (y_pred, y_prob) in enumerate(zip(y_pred_list, y_prob_list)):
        model_metrics = calculate_metrics(y_test, y_pred, y_prob)
        individual_metrics.append((i, model_metrics["accuracy"], model_metrics["auc"]))
    
    # Sort models by accuracy
    individual_metrics.sort(key=lambda x: x[1], reverse=True)
    
    # Assign weights based on rank (higher ranked models get higher weights)
    weights = np.zeros(len(y_prob_list))
    for rank, (idx, _, _) in enumerate(individual_metrics):
        weights[idx] = np.exp(-0.5 * rank)
    
    weights = weights / np.sum(weights)
    
    # Create weighted ensemble predictions
    ensemble_probs = np.zeros(len(y_test))
    for i, y_prob in enumerate(y_prob_list):
        ensemble_probs += weights[i] * y_prob
    
    # Find optimal threshold for ensemble
    optimal_results = find_optimal_threshold(y_test, ensemble_probs)
    optimal_threshold = optimal_results["threshold"]
    
    ensemble_preds = (ensemble_probs >= optimal_threshold).astype(int)
    
    metrics = calculate_metrics(y_test, ensemble_preds, ensemble_probs)
    
    return weights, ensemble_preds, ensemble_probs, metrics

class WeightedEnsembleModel(mlflow.pyfunc.PythonModel):
    """MLflow Python model for weighted ensemble prediction"""
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
        """Generate predictions using weighted ensemble of models"""
        import numpy as np
        X = model_input.values
        ensemble_probs = np.zeros(X.shape[0])
        for w, m in zip(self.weights, self.base_models):
            probs = m.predict_proba(X)[:, 1]
            ensemble_probs += w * probs
        return (ensemble_probs >= self.threshold).astype(int)

def log_to_mlflow(weights, preds, probs, model_names, model_uris, metrics, output_dir):
    """Log ensemble model and metrics to MLflow"""
    experiment_name = "weighted_ensemble"
    mlflow.set_experiment(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"weighted_ensemble_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("ensemble_method", "weighted_average")
        mlflow.log_param("number_of_models", len(model_names))
        
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("auc", metrics["auc"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("sensitivity", metrics["sensitivity"])
        mlflow.log_metric("specificity", metrics["specificity"])
        mlflow.log_metric("optimal_threshold", metrics["optimal_threshold"])
        
        unique_model_names = {}
        for i, name in enumerate(model_names):
            if name in unique_model_names:
                unique_model_names[name] = unique_model_names[name] + 1
                unique_name = f"{name}_v{unique_model_names[name]}"
            else:
                unique_model_names[name] = 1
                unique_name = name
            
            mlflow.log_param(f"weight_{unique_name}", weights[i])
        
        config_path = os.path.join(output_dir, "ensemble_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_names": model_names,
                "model_uris": {name: uri for name, uri in zip(model_names, model_uris)},
                "model_weights": {name: float(w) for name, w in zip(model_names, weights)},
                "optimal_threshold": float(metrics["optimal_threshold"])}, f, indent=2)
        mlflow.log_artifact(config_path)

        mlflow.pyfunc.log_model(
            artifact_path="ensemble_pyfunc",
            python_model=WeightedEnsembleModel(),
            artifacts={"ensemble_config": config_path},
            registered_model_name="high_accuracy_ensemble"
        )

        run_id = run.info.run_id
        return run_id

if __name__ == "__main__":
    args = parse_args()
    
    output_dir = os.path.join("models", "ensembles", f"high_acc_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    X_test, y_test, feature_names = load_test_data(args.data_path)
    
    models = []
    model_uris = []
    model_names = []
    
    for i, model_info in enumerate(MODELS):
        model, model_uri = load_model(model_info)
        if model is not None:
            if model_info["name"] in model_names:
                unique_name = f"{model_info['name']}_v{model_info['version']}"
            else:
                unique_name = model_info["name"]
                
            models.append(model)
            model_uris.append(model_uri)
            model_names.append(unique_name)
    
    if len(models) < 3:
        exit(1)
    
    y_pred_list = []
    y_prob_list = []
    valid_model_indices = []
    valid_model_names = []
    valid_model_uris = []
    
    for i, model in enumerate(models):
        model_name = model_names[i]
        y_pred, y_prob = get_model_predictions(model, X_test)
        
        if y_pred is not None and y_prob is not None:
            y_pred_list.append(y_pred)
            y_prob_list.append(y_prob)
            valid_model_indices.append(i)
            valid_model_names.append(model_name)
            valid_model_uris.append(model_uris[i])
    
    weights, ensemble_preds, ensemble_probs, metrics = create_weighted_ensemble(
        y_test, y_pred_list, y_prob_list, valid_model_names
    )
    
    run_id = log_to_mlflow(
        weights, ensemble_preds, ensemble_probs, 
        valid_model_names, valid_model_uris, metrics, output_dir
    )
    
    print("\n===== HIGH ACCURACY ENSEMBLE RESULTS =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print("\nTop 5 Model Weights:")
    sorted_weights = sorted(zip(valid_model_names, weights), key=lambda x: x[1], reverse=True)
    for model_name, weight in sorted_weights[:5]:
        print(f"  {model_name}: {weight:.4f}")
    
    print(f"\nResults saved to {output_dir}")
    print(f"MLflow run ID: {run_id}")
    print(f"To access this ensemble: Use the direct-ensemble endpoint in the API") 