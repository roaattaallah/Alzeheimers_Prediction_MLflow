import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import random
import joblib
import mlflow
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import subprocess
from hyperopt import fmin, tpe, STATUS_OK, Trials
import shutil 
import json
from datetime import datetime
import seaborn as sns
from mlflow_config import setup_mlflow


# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)


setup_mlflow()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
mlflow.set_experiment("tuned_models")

def load_data():
    """Load the preprocessed data."""
    processed_dir = os.path.join("data", "processed")
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    feature_names = pd.read_csv(os.path.join(processed_dir, 'feature_names.csv'))['0'].values
    
    return X_train, X_test, y_train, y_test, feature_names

def import_module_from_file(module_name, file_path):
    """Import Python file as module dynamically."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def direct_train_model(model_type):
    """Train a model directly by importing train.py as a module"""
    logger.info(f"Training model: {model_type}")
    
    if mlflow.active_run():
        mlflow.end_run()
    
    train_module = import_module_from_file("train_module", "train.py")
    
    class CaptureMetrics:
        def __init__(self):
            self.metrics = {}
        
        def log_metric(self, name, value):
            self.metrics[name] = value
            logger.info(f"{name}: {value:.4f}")
    
    metrics_capture = CaptureMetrics()
    original_info = train_module.logger.info
    
    # Override logger.info to capture metrics from output
    def new_info(message):
        original_info(message)
        if ": " in message:
            parts = message.split(": ")
            if len(parts) == 2:
                try:
                    name = parts[0].strip()
                    value = float(parts[1].strip())
                    if name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                        metrics_capture.metrics[name] = value
                except:
                    pass
    
    train_module.logger.info = new_info
    
    # Create mock args for train.py
    class Args:
        def __init__(self, model_type):
            self.model_type = model_type
            self.alpha = 0.5
            self.l1_ratio = 0.1
            self.use_best_params = "False"
    
    train_module.parse_args = lambda: Args(model_type)
    
    try:
        train_module.main()
        accuracy = metrics_capture.metrics.get("accuracy", 0.0)
        logger.info(f"{model_type} model accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}
    except Exception as e:
        logger.error(f"Error training {model_type} model: {str(e)}")
        return {"accuracy": 0.0}
    finally:
        train_module.logger.info = original_info
        if mlflow.active_run():
            mlflow.end_run()

def direct_tune_hyperparameters(model_type):
    """Tune hyperparameters directly by importing hyperparameter_tuning.py as a module"""
    logger.info(f"Tuning hyperparameters for: {model_type}")
    
    if mlflow.active_run():
        mlflow.end_run()
    
    tuning_module = import_module_from_file("tuning_module", "hyperparameter_tuning.py")
    
    class CaptureMetrics:
        def __init__(self):
            self.best_score = 0.0
        
        def log_metric(self, name, value):
            if name == "best_score" or name == "test_roc_auc":
                self.best_score = max(self.best_score, value)
                logger.info(f"{name}: {value:.4f}")
    
    metrics_capture = CaptureMetrics()
    original_info = tuning_module.logger.info
    
    # Override logger.info to capture metrics from output
    def new_info(message):
        original_info(message)
        if "Best score: " in message:
            try:
                value = float(message.split("Best score: ")[1].strip())
                metrics_capture.best_score = value
            except:
                pass
        elif "Test roc_auc: " in message:
            try:
                value = float(message.split("Test roc_auc: ")[1].strip())
                metrics_capture.best_score = max(metrics_capture.best_score, value)
            except:
                pass
    
    tuning_module.logger.info = new_info
    
    # Create mock args for hyperparameter_tuning.py
    class Args:
        def __init__(self, model_type):
            self.model_type = model_type
            self.max_evals = 50
    
    tuning_module.parse_args = lambda: Args(model_type)
    
    try:
        tuning_module.main()
        best_score = metrics_capture.best_score
        logger.info(f"{model_type} model best score: {best_score:.4f}")
        return {"best_score": best_score}
    except Exception as e:
        logger.error(f"Error tuning {model_type} model: {str(e)}")
        return {"best_score": 0.0}
    finally:
        tuning_module.logger.info = original_info
        if mlflow.active_run():
            mlflow.end_run()

def train_base_models():
    """Train all base models and return their performance metrics."""
    model_types = ["logistic", "rf", "knn", "svm", "xgboost", "nn"]
    results = {}
    
    for model_type in model_types:
        try:
            model_results = direct_train_model(model_type)
            results[model_type] = model_results
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")
    
    return results

def tune_hyperparameters(model_types=None):
    """Run hyperparameter tuning for specified models."""
    if model_types is None:
        model_types = ["logistic", "rf", "knn", "svm", "xgboost", "nn"]
    
    tuning_results = {}
    for model_type in model_types:
        try:
            tuning_result = direct_tune_hyperparameters(model_type)
            tuning_results[model_type] = tuning_result
        except Exception as e:
            logger.error(f"Error tuning {model_type} model: {str(e)}")
    
    return tuning_results

def find_best_model(tuning_results):
    """Find the best model based on hyperparameter tuning results."""
    if not tuning_results:
        logger.error("No tuning results available")
        return None
        
    best_model = max(tuning_results.items(), key=lambda x: x[1].get("best_score", 0))
    logger.info(f"Best model: {best_model[0]} with score: {best_model[1].get('best_score', 0):.4f}")
    return best_model[0]

def load_best_params(model_type):
    """Load the best hyperparameters for a model type."""
    params_path = os.path.join("models", "tuned_models", f"{model_type}_best_params.json")
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading best parameters: {str(e)}")
    return {}

def train_and_save_best_model(best_model_type):
    """Train and save the best model with optimal hyperparameters."""
    if best_model_type is None:
        logger.error("No best model type specified")
        return
    
    best_params = load_best_params(best_model_type)
    logger.info(f"Training final model: {best_model_type}")
    
    try:
        # Train the best model with its optimal parameters
        cmd = ["python", "train.py", "--model_type", best_model_type, "--use_best_params", "True"]
        subprocess.run(cmd, check=True)
        
        # Save the best model to the compared_models directory
        os.makedirs(os.path.join("models", "compared_models"), exist_ok=True)
        best_model_path = os.path.join("models", "trained_models", f"{best_model_type}_model.pkl")
        best_model_dest = os.path.join("models", "compared_models", "best_model.pkl")
        
        if os.path.exists(best_model_dest):
            os.remove(best_model_dest)
        shutil.copy2(best_model_path, best_model_dest)
        
        # Save model metadata
        model_info = {
            "model_type": best_model_type,
            "hyperparameters": best_params,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        model_info_path = os.path.join("models", "compared_models", "best_model_info.json")
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Best model saved to {best_model_dest}")
        
        # Log best model to MLflow
        run = None
        try:
            run = mlflow.start_run(run_name=f"best_model_{best_model_type}")
            mlflow.log_param("model_type", best_model_type)
            
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            mlflow.log_artifact(model_info_path)
            mlflow.log_artifact(best_model_dest)
            
            registered_model_name = f"alzheimers_best_model_{best_model_type}"
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{os.path.basename(best_model_dest)}",
                registered_model_name
            )
            
            mlflow.set_tag("status", "SUCCESS")
        finally:
            if run:
                mlflow.end_run()
            
    except Exception as e:
        logger.error(f"Error training final model: {str(e)}")

def generate_comparison_chart(base_results, tuned_results):
    """Generate a comparison chart of base and tuned models."""
    models = []
    base_scores = []
    tuned_scores = []
    improvements = []
    
    sorted_results = sorted(
        base_results.items(),
        key=lambda x: x[1].get('accuracy', 0),
        reverse=True
    )
    
    if not tuned_results:
        logger.warning("No tuning results available, using only base model results")
        for model_name, base_metrics in sorted_results:
            base_accuracy = base_metrics.get('accuracy', 0)
            if base_accuracy == 0:
                continue
            models.append(model_name)
            base_scores.append(base_accuracy)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, base_scores, color='skyblue')
        
        for bar in bars:
            height = bar.get_height()
            plt.annotate('{:.3f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        plt.ylabel('Accuracy')
        plt.title('Base Model Performance Comparison')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        os.makedirs("reports/figures", exist_ok=True)
        fig_path = "reports/figures/model_comparison.png"
        plt.savefig(fig_path)
        plt.close()
        
        best_model = models[0] if models else "logistic"
        return best_model
    
    # Create comparison chart for base vs tuned models
    for model_name, base_metrics in sorted_results:
        base_accuracy = base_metrics.get('accuracy', 0)
        tuned_accuracy = tuned_results.get(model_name, {}).get('best_score', 0)
        
        if base_accuracy == 0 or tuned_accuracy == 0:
            continue
        
        models.append(model_name)
        base_scores.append(base_accuracy)
        tuned_scores.append(tuned_accuracy)
        improvements.append(tuned_accuracy - base_accuracy)
    
    if not models:
        logger.warning("No valid models with both scores, using best base model")
        best_base_model = sorted_results[0][0] if sorted_results else "logistic"
        return best_base_model
    
    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    x = range(len(models))
    width = 0.35
    
    base_bars = ax1.bar([i - width/2 for i in x], base_scores, width, label='Base Model', color='skyblue')
    tuned_bars = ax1.bar([i + width/2 for i in x], tuned_scores, width, label='Tuned Model', color='orange')
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.annotate('{:.3f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    add_labels(base_bars)
    add_labels(tuned_bars)
    
    ax1.set_ylabel('Accuracy / Score')
    ax1.set_title('Model Performance Comparison: Base vs. Tuned')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot improvement bars
    improvement_bars = ax2.bar(x, improvements, width=0.5, color='green', alpha=0.6)
    
    for bar in improvement_bars:
        height = bar.get_height()
        ax2.annotate('{:+.3f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax2.set_ylabel('Improvement')
    ax2.set_title('Performance Improvement from Hyperparameter Tuning')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/model_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    
    # Return the best tuned model
    best_idx = tuned_scores.index(max(tuned_scores))
    best_model = models[best_idx]
    
    return best_model

def clear_model_folders():
    """Clear the model folders to ensure clean state."""
    valid_folders = ["trained_models", "tuned_models", "compared_models"]
    
    for folder_name in valid_folders:
        folder_path = os.path.join("models", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        if os.path.exists(folder_path):
            logger.info(f"Clearing folder: {folder_path}")
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")

def main():
    """Main function to run the comparison process."""
    logger.info("Starting model comparison process")
    
    # Clean up model folders
    clear_model_folders()
    
    # Train base models
    logger.info("Training base models...")
    base_results = train_base_models()
    
    # Tune hyperparameters
    logger.info("Running hyperparameter tuning...")
    tuned_results = tune_hyperparameters()
    
    # Generate comparison chart and find best model
    logger.info("Generating comparison chart...")
    best_model_type = generate_comparison_chart(base_results, tuned_results)
    
    logger.info(f"Best model type: {best_model_type}")
    logger.info("Training and saving best model...")
    train_and_save_best_model(best_model_type)
    
    logger.info("Model comparison process completed")

if __name__ == "__main__":
    main() 