#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import random
import mlflow
from mlflow.tracking import MlflowClient
import logging
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# Import MLflow configuration
from mlflow_config import setup_mlflow
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Set up MLflow tracking
setup_mlflow()

# Check for optional packages
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential #type:ignore
    from tensorflow.keras.layers import Dense, Dropout #type:ignore
    tf.random.set_seed(RANDOM_SEED)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for hyperopt
X_train = None
y_train = None

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning script")
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=["logistic", "rf", "knn", "svm", "xgboost", "nn"],
        help="Type of model to tune",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=50,
        help="Maximum number of evaluations for hyperparameter tuning",
    )
    return parser.parse_args()

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def remove_existing_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def load_data():
    processed_dir = os.path.join("data", "processed")
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    return X_train, y_train

def evaluate_final_model(model, X_test, y_test):
    """
    Evaluate a model on the test set and return metrics.
    
    Parameters:
    -----------
    model : trained model object
        The trained model to evaluate.
    X_test : array-like
        The test features.
    y_test : array-like
        The test target values.
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics.
    """
    # Handle TensorFlow models
    if hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
        # TensorFlow model
        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        # scikit-learn model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    return metrics

def log_params_and_metrics(params, metric_value, model_type, run_name=None):
    """Centralized function for logging parameters and metrics to MLflow"""
    nested_run = mlflow.start_run(nested=True, run_name=run_name)
    try:
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_metric("mean_cv_auc", metric_value)
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        logger.info(f"{model_type} with {param_str}: mean AUC = {metric_value:.4f}")
    finally:
        mlflow.end_run()

def objective_logistic(params):
    C = params['C']
    l1_ratio = params['l1_ratio']
    
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=C,
        l1_ratio=l1_ratio,
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    mean_auc = cv_scores.mean()
    
    log_params_and_metrics(
        {'C': C, 'l1_ratio': l1_ratio}, 
        mean_auc,
        'Logistic Regression',
        f"logistic_trial_C{C:.4f}_l1{l1_ratio:.2f}"
    )
    
    return {'loss': -mean_auc, 'status': STATUS_OK, 'model': model}

def objective_random_forest(params):
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_samples_split = int(params['min_samples_split'])
    min_samples_leaf = int(params['min_samples_leaf'])
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',  
        random_state=RANDOM_SEED
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    mean_auc = cv_scores.mean()
    
    log_params_and_metrics(
        {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        },
        mean_auc,
        'Random Forest',
        f"rf_trial_{n_estimators}trees_d{max_depth}"
    )
    
    return {'loss': -mean_auc, 'status': STATUS_OK, 'model': model}

def objective_knn(params):
    n_neighbors = int(params['n_neighbors'])
    weights = params['weights']
    p = int(params['p'])
    leaf_size = int(params['leaf_size'])
    
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        leaf_size=leaf_size,
        algorithm='auto'
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    mean_auc = cv_scores.mean()
    
    log_params_and_metrics(
        {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'p': p,
            'leaf_size': leaf_size
        },
        mean_auc,
        'KNN'
    )
    
    return {'loss': -mean_auc, 'status': STATUS_OK, 'model': model}

def objective_svm(params):
    C = params['C']
    gamma = params['gamma']
    
    model = SVC(
        C=C,
        gamma=gamma,
        class_weight='balanced',
        probability=True,
        random_state=RANDOM_SEED
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    mean_auc = cv_scores.mean()
    
    log_params_and_metrics({'C': C, 'gamma': gamma}, mean_auc, 'SVM')
    
    return {'loss': -mean_auc, 'status': STATUS_OK, 'model': model}

def objective_xgboost(params):
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost is not installed. Please install it with: pip install xgboost")
        return {'loss': 0, 'status': STATUS_OK, 'model': None}
    
    n_estimators = int(params['n_estimators'])
    learning_rate = params['learning_rate']
    max_depth = int(params['max_depth'])
    subsample = float(params['subsample'])
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=RANDOM_SEED
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    mean_auc = cv_scores.mean()
    
    log_params_and_metrics(
        {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample
        },
        mean_auc,
        'XGBoost'
    )
    
    return {'loss': -mean_auc, 'status': STATUS_OK, 'model': model}

def create_neural_network(input_dim, units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(units, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units//2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def objective_neural_network(params):
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow is not installed")
        return {'loss': 0, 'status': STATUS_OK, 'model': None}
    
    units = int(params['units'])
    dropout_rate = float(params['dropout_rate'])
    learning_rate = float(params['learning_rate'])
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])
    
    input_dim = X_train.shape[1]
    
    # K-Fold Cross-Validation
    k_folds = 3
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []
    
    nested_run = mlflow.start_run(nested=True, run_name=f"nn_trial_u{units}_dr{dropout_rate}_lr{learning_rate}")
    
    try:
        mlflow.log_param("units", units)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model = create_neural_network(
                input_dim=input_dim,
                units=units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            model.fit(
                X_fold_train, y_fold_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_fold_val, y_fold_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            y_pred_proba = model.predict(X_fold_val)
            auc_score = roc_auc_score(y_fold_val, y_pred_proba)
            fold_scores.append(auc_score)
            
            mlflow.log_metric(f"fold_{fold+1}_auc", auc_score)
            
            tf.keras.backend.clear_session()
        
        mean_auc = np.mean(fold_scores)
        mlflow.log_metric("mean_cv_auc", mean_auc)
        
        logger.info(f"Neural Network with units={units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}, "
                   f"batch_size={batch_size}, epochs={epochs}: mean AUC = {mean_auc:.4f}")
        
        # Final model trained on all data
        final_model = create_neural_network(
            input_dim=input_dim,
            units=units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        final_model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )],
            verbose=0
        )
        
        return {'loss': -mean_auc, 'status': STATUS_OK, 'model': final_model}
    
    except Exception as e:
        logger.error(f"Error in neural network training: {str(e)}")
        return {'loss': 0, 'status': STATUS_OK, 'model': None}
    
    finally:
        mlflow.end_run()
        tf.keras.backend.clear_session()

def save_best_params(best_params, model_type):
    ensure_directory_exists("models")
    ensure_directory_exists(os.path.join("models", "tuned_models"))
    
    best_params_path = os.path.join("models", "tuned_models", f"{model_type}_best_params.json")
    remove_existing_file(best_params_path)
    
    # Convert hyperopt choices to actual parameter values
    clean_params = {}
    
    # Define the actual parameter value lists for each parameter
    param_values = {
        # Logistic regression
        'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        
        # Random forest
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [3, 5, 7, 10, 12, 15],
        'min_samples_split': [2, 3, 5, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        
        # KNN
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'leaf_size': [20, 30, 40, 50],
        
        # SVM
        'gamma': [0.001, 0.01, 0.1, 0.5, 1.0],
        
        # XGBoost
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        
        # Neural Network
        'units': [64, 128, 256],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [32, 64, 128],
        'epochs': [30, 50, 100]
    }
    
    # Convert indices to actual values
    for key, index in best_params.items():
        if hasattr(index, 'item'):
            index = index.item()  # Convert numpy values to Python native types
        
        # For max_evals=1, just use fixed values
        if key == 'units' and index == 0:
            clean_params[key] = 256
            logger.info(f"Converted parameter {key}: index {index} -> value {clean_params[key]}")
        elif key == 'dropout_rate' and index == 0:
            clean_params[key] = 0.4
            logger.info(f"Converted parameter {key}: index {index} -> value {clean_params[key]}")
        elif key == 'learning_rate' and index == 0:
            clean_params[key] = 0.005
            logger.info(f"Converted parameter {key}: index {index} -> value {clean_params[key]}")
        elif key == 'batch_size' and index == 0:
            clean_params[key] = 64
            logger.info(f"Converted parameter {key}: index {index} -> value {clean_params[key]}")
        elif key == 'epochs' and index == 0:
            clean_params[key] = 30
            logger.info(f"Converted parameter {key}: index {index} -> value {clean_params[key]}")
        # If the key exists in param_values, convert index to actual value
        elif key in param_values and isinstance(index, (int, float)) and index < len(param_values[key]):
            clean_params[key] = param_values[key][int(index)]
            logger.info(f"Converted parameter {key}: index {index} -> value {clean_params[key]}")
        else:
            clean_params[key] = index
    
    try:
        with open(best_params_path, 'w') as f:
            import json
            json.dump(clean_params, f, indent=2)
        logger.info(f"Saved best parameters to {best_params_path}: {clean_params}")
    except Exception as e:
        logger.error(f"Error saving best parameters: {str(e)}")

def get_search_space(model_type, max_evals):
    """Get search space for a model type based on max_evals"""
    if max_evals == 1:
        # For faster testing, use fixed hyperparameters
        if model_type == "logistic":
            return {
                'C': hp.choice('C', [2.0]),
                'l1_ratio': hp.choice('l1_ratio', [0.3])
            }
        elif model_type == "rf":
            return {
                'n_estimators': hp.choice('n_estimators', [200]),
                'max_depth': hp.choice('max_depth', [12]),
                'min_samples_split': hp.choice('min_samples_split', [5]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [2])
            }
        elif model_type == "knn":
            return {
                'n_neighbors': hp.choice('n_neighbors', [9]),
                'weights': hp.choice('weights', ['distance']),
                'p': hp.choice('p', [2]),
                'leaf_size': hp.choice('leaf_size', [42])
            }
        elif model_type == "svm":
            return {
                'C': hp.choice('C', [0.5]),
                'gamma': hp.choice('gamma', [0.5])
            }
        elif model_type == "xgboost":
            return {
                'n_estimators': hp.choice('n_estimators', [56]),
                'learning_rate': hp.choice('learning_rate', [0.2]),
                'max_depth': hp.choice('max_depth', [5]),
                'subsample': hp.choice('subsample', [1.0])
            }
        elif model_type == "nn":
            return {
                'units': hp.choice('units', [256]),
                'dropout_rate': hp.choice('dropout_rate', [0.4]),
                'learning_rate': hp.choice('learning_rate', [0.005]),
                'batch_size': hp.choice('batch_size', [64]),
                'epochs': hp.choice('epochs', [30])
            }
    
    # Normal search spaces for full tuning
    if model_type == "logistic":
        return {
            'C': hp.choice('C', [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
            'l1_ratio': hp.choice('l1_ratio', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        }
    elif model_type == "rf":
        return {
            'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200, 250, 300]),
            'max_depth': hp.choice('max_depth', [3, 5, 7, 10, 12, 15]),
            'min_samples_split': hp.choice('min_samples_split', [2, 3, 5, 8, 10]),
            'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 3, 4, 5])
        }
    elif model_type == "knn":
        return {
            'n_neighbors': hp.choice('n_neighbors', [3, 5, 7, 9, 11, 13, 15]),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'p': hp.choice('p', [1, 2]),
            'leaf_size': hp.choice('leaf_size', [20, 30, 40, 50])
        }
    elif model_type == "svm":
        return {
            'C': hp.choice('C', [0.1, 0.5, 1.0, 2.0, 5.0]),
            'gamma': hp.choice('gamma', [0.001, 0.01, 0.1, 0.5, 1.0])
        }
    elif model_type == "xgboost":
        return {
            'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200]),
            'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.1, 0.2]),
            'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
            'subsample': hp.choice('subsample', [0.7, 0.8, 0.9, 1.0])
        }
    elif model_type == "nn":
        return {
            'units': hp.choice('units', [64, 128, 256]),
            'dropout_rate': hp.choice('dropout_rate', [0.2, 0.3, 0.4]),
            'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.01]),
            'batch_size': hp.choice('batch_size', [32, 64, 128]),
            'epochs': hp.choice('epochs', [30, 50, 100])
        }
    
    # Default empty search space
    return {}

def save_model(model, model_type, X_train, y_train, feature_names):
    """Save the model to disk and MLflow"""
    # 1. Create local directories for filesystem storage
    ensure_directory_exists("models")
    ensure_directory_exists(os.path.join("models", "tuned_models"))
    
    # Local filesystem paths
    local_model_path = os.path.join("models", "tuned_models", f"{model_type}_best_model.pkl")
    remove_existing_file(local_model_path)
    
    # 2. Define MLflow artifact paths - this determines the folder structure in MLflow UI
    # Store directly in the run's artifacts, not in a nested folder
    model_artifact_path = f"{model_type}_best_model"
    
    # Sample for model signature
    sample_df = pd.DataFrame(X_train[:3], columns=feature_names)
    
    # 3. Save model based on type
    if model_type == "nn" and TENSORFLOW_AVAILABLE:
        # TensorFlow model - local filesystem save
        local_tf_path = os.path.join("models", "tuned_models", f"{model_type}_best_model.keras")
        remove_existing_file(local_tf_path)
        model.save(local_tf_path)
        
        # Log TensorFlow model to MLflow with signature
        try:
            # Create input example for TensorFlow model
            input_example = X_train[:3]
            
            # Log model with input example
            mlflow.tensorflow.log_model(
                model,
                artifact_path=model_artifact_path,
                input_example=input_example
            )
            logger.info(f"Saved TensorFlow model to MLflow at: {model_artifact_path}")
        except Exception as e:
            logger.error(f"Error logging TensorFlow model: {str(e)}")
            # Fallback: log the saved model file as an artifact
            mlflow.log_artifact(local_tf_path)
            logger.info(f"Saved TensorFlow model file to MLflow as: {os.path.basename(local_tf_path)}")
    else:
        # Sklearn model - local filesystem save
        joblib.dump(model, local_model_path)
        
        # Log sklearn model to MLflow
        mlflow.sklearn.log_model(
            model, 
            artifact_path=model_artifact_path,
            input_example=sample_df
        )
        logger.info(f"Saved sklearn model to MLflow at: {model_artifact_path}")
        
        # Also log the saved model file as an artifact for direct access
        mlflow.log_artifact(local_model_path)
    
    # 4. Log the best params file if it exists
    best_params_path = os.path.join("models", "tuned_models", f"{model_type}_best_params.json")
    if os.path.exists(best_params_path):
        mlflow.log_artifact(best_params_path)
        logger.info(f"Saved parameter file to MLflow as: {os.path.basename(best_params_path)}")
    
    return model_artifact_path

def get_base_model_metrics(model_type):
    """Get metrics from the base model stored in MLflow"""
    client = mlflow.tracking.MlflowClient()
    
    # Try to find the base model
    base_model_name = f"alzheimers_{model_type}_model"
    
    try:
        # Get the latest version of the base model
        versions = client.search_model_versions(f"name='{base_model_name}'")
        if not versions:
            logger.warning(f"No base model found for {model_type}")
            return {"roc_auc": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        # Sort by version and get the latest
        versions.sort(key=lambda x: int(x.version), reverse=True)
        latest_version = versions[0]
        
        # Get the run ID for this model version
        run_id = latest_version.run_id
        
        # Get the metrics for this run
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        return {
            "roc_auc": metrics.get("roc_auc", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0)
        }
    except Exception as e:
        logger.warning(f"Error getting base model metrics: {str(e)}")
        return {"roc_auc": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

def create_comparison_plot(base_metrics, tuned_metrics, model_type):
    """Create a plot comparing all metrics between base and tuned models"""
    plt.figure(figsize=(12, 8))
    
    # Metrics to include
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    # Number of metrics
    n_metrics = len(metrics)
    
    # Width of bars
    bar_width = 0.35
    
    # Positions for bars
    r1 = np.arange(n_metrics)
    r2 = [x + bar_width for x in r1]
    
    # Extract metric values
    base_values = [base_metrics.get(metric, 0) for metric in metrics]
    tuned_values = [tuned_metrics.get(metric, 0) for metric in metrics]
    
    # Create grouped bars
    bars1 = plt.bar(r1, base_values, width=bar_width, label='Base Model', color='blue', alpha=0.7)
    bars2 = plt.bar(r2, tuned_values, width=bar_width, label='Tuned Model', color='green', alpha=0.7)
    
    # Add values on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Calculate and display improvements
    for i, (metric, base_val, tuned_val) in enumerate(zip(metrics, base_values, tuned_values)):
        improvement = tuned_val - base_val
        plt.annotate(f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}", 
                     xy=(r2[i], tuned_val),
                     xytext=(r2[i] + 0.1, tuned_val + 0.03),
                     fontsize=9,
                     color='green' if improvement > 0 else 'red')
    
    # Customize plot
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title(f'{model_type.upper()} Model Performance Comparison')
    plt.xticks([r + bar_width/2 for r in range(n_metrics)], metrics_labels)
    plt.ylim(0, 1.2)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    output_path = os.path.join("models", "tuned_models", f"{model_type}_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    args = parse_args()
 
    # Load data
    logger.info("Loading preprocessed data...")
    global X_train, y_train
    X_train, y_train = load_data()
    
    # Load test data
    processed_dir = os.path.join("data", "processed")
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    # Load feature names
    feature_names = pd.read_csv(os.path.join(processed_dir, 'feature_names.csv'))['0'].values
    
    # Get search space and objective function
    search_space = get_search_space(args.model_type, args.max_evals)
    
    if args.model_type == "logistic":
        objective = objective_logistic
    elif args.model_type == "rf":
        objective = objective_random_forest
    elif args.model_type == "knn":
        objective = objective_knn
    elif args.model_type == "svm":
        objective = objective_svm
    elif args.model_type == "xgboost":
        objective = objective_xgboost
    elif args.model_type == "nn":
        objective = objective_neural_network
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
            return
    
    # Set MLflow experiment
    experiment_name = "tuned_models"
    mlflow.set_experiment(experiment_name)
    
    # Model name
    model_name = f"{args.model_type}_tuned"
    
    # Log the MLflow structure for clarity
    logger.info(f"MLflow structure: experiment '{experiment_name}' → run '{model_name}'")
    
    # Clean up previous runs
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{model_name}'"
        )
        for run in runs:
            client.delete_run(run.info.run_id)
            logger.info(f"Deleted previous run with name '{model_name}'")
    
    # Start MLflow parent run
    run = None
    try:
        run = mlflow.start_run(run_name=model_name)
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("max_evals", args.max_evals)
        
        # Run hyperparameter optimization
        logger.info(f"Starting hyperparameter tuning for {args.model_type} model")
        trials = Trials()
        
        # Create a deterministic search algorithm
        from hyperopt import rand
        algo = rand.suggest
        
        best = fmin(
            fn=objective,
            space=search_space,
            algo=algo,
            max_evals=args.max_evals,
            trials=trials
        )
        
        # Get the best model
        best_trial_idx = np.argmin([trial['result']['loss'] for trial in trials.trials])
        best_model = trials.trials[best_trial_idx]['result']['model']
        
        # Log best parameters
        logger.info(f"Best hyperparameters: {best}")
        for param_name, param_value in best.items():
                mlflow.log_param(f"best_{param_name}", param_value)
        
        # Evaluate on test set
        best_model.fit(X_train, y_train)
        metrics = evaluate_final_model(best_model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Save model
        model_artifact_path = save_model(best_model, args.model_type, X_train, y_train, feature_names)
        
        # Mark as successful
        mlflow.set_tag("status", "SUCCESS")

        # Create and log the comparison visualization
        base_metrics = get_base_model_metrics(args.model_type)
        visualization_path = create_comparison_plot(base_metrics, metrics, args.model_type)
        
        # Log the visualization
        mlflow.log_artifact(visualization_path)
        logger.info(f"Created and logged model comparison visualization")

        # Register model
        try:
            client = mlflow.tracking.MlflowClient()
            registered_model_name = f"alzheimers_{args.model_type}_tuned"
            
            model_details = mlflow.register_model(
                f"runs:/{run.info.run_id}/{model_artifact_path}",
                registered_model_name
            )
            
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_details.version,
                stage="Production"
            )
            
            mlflow.log_param("model_version", model_details.version)
            logger.info(f"Registered model {registered_model_name} version {model_details.version}")
            logger.info(f"Model artifacts saved to: {model_artifact_path}")
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")

        # Save best parameters
        save_best_params(best, args.model_type)
    finally:
        if run:
            mlflow.end_run()

if __name__ == "__main__":
    main() 