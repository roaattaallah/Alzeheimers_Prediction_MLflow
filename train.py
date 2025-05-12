#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import random
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import glob
import re
import warnings
warnings.filterwarnings('ignore')

# Import MLflow configuration
from mlflow_config import setup_mlflow

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Check if XGBoost is available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Check if TensorFlow/Keras is available
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Set up MLflow tracking
setup_mlflow()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Model training script")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Regularization strength parameter for Logistic Regression (default: 0.5)",
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.1,
        help="L1 ratio parameter for Logistic Regression with elastic net (default: 0.1)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=["logistic", "rf", "knn", "svm", "xgboost", "nn"],
        help="Type of model to train (default: logistic, choices: logistic, rf, knn, svm, xgboost, nn)",
    )
    parser.add_argument(
        "--use_best_params",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether to use the best hyperparameters from tuning (default: False)",
    )
    return parser.parse_args()

def ensure_directory_exists(directory):
    """
    Ensure a directory exists, create it if it doesn't.
    
    Parameters:
    -----------
    directory : str
        Directory path to check or create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def remove_existing_file(file_path):
    """
    Remove a file if it exists to ensure overwriting.
    
    Parameters:
    -----------
    file_path : str
        Path to the file to remove
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Removed existing file: {file_path}")

def load_data():
    """
    Load the preprocessed data.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy arrays
        The preprocessed training and testing data.
    """
    processed_dir = os.path.join("data", "processed")
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    # Load feature names
    feature_names = pd.read_csv(os.path.join(processed_dir, 'feature_names.csv'))['0'].values
    
    return X_train, X_test, y_train, y_test, feature_names

def train_logistic_regression(X_train, y_train, alpha, l1_ratio):
    """
    Train a logistic regression model.
    
    Parameters:
    -----------
    X_train : numpy array
        The training features.
    y_train : numpy array
        The training target.
    alpha : float
        The regularization strength.
    l1_ratio : float
        The L1 ratio for elastic net regularization.
        
    Returns:
    --------
    model : LogisticRegression
        The trained logistic regression model.
    """
    logger.info(f"Training Logistic Regression with alpha={alpha}, l1_ratio={l1_ratio}")
    
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=1.0/alpha,
        l1_ratio=l1_ratio,
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
    
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a random forest classifier.
    
    Parameters:
    -----------
    X_train : numpy array
        The training features.
    y_train : numpy array
        The training target.
        
    Returns:
    --------
    model : RandomForestClassifier
        The trained random forest model.
    """
    logger.info("Training Random Forest Classifier")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED
    )
    
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    """
    Train a K-Nearest Neighbors classifier.
    
    Parameters:
    -----------
    X_train : numpy array
        The training features.
    y_train : numpy array
        The training target.
        
    Returns:
    --------
    model : KNeighborsClassifier
        The trained K-Nearest Neighbors model.
    """
    logger.info("Training K-Nearest Neighbors Classifier")
    
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2  # Euclidean distance
    )
    
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine classifier.
    
    Parameters:
    -----------
    X_train : numpy array
        The training features.
    y_train : numpy array
        The training target.
        
    Returns:
    --------
    model : SVC
        The trained SVM model.
    """
    logger.info("Training Support Vector Machine Classifier")
    
    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=RANDOM_SEED
    )
    
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier.
    
    Parameters:
    -----------
    X_train : numpy array
        The training features.
    y_train : numpy array
        The training target.
        
    Returns:
    --------
    model : XGBClassifier
        The trained XGBoost model.
    """
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost is not installed. Please install it with 'pip install xgboost'")
        raise ImportError("XGBoost is not installed")
        
    logger.info("Training XGBoost Classifier")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_SEED
    )
    
    model.fit(X_train, y_train)
    return model

def create_neural_network(input_dim, units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    Create a neural network model for binary classification.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features.
    units : int
        Number of units in the hidden layer.
    dropout_rate : float
        Dropout rate for regularization.
    learning_rate : float
        Learning rate for the optimizer.
        
    Returns:
    --------
    model : Sequential
        The compiled neural network model.
    """
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow is not installed. Please install it to use neural networks.")
        return None
        
    # Import inside function to avoid errors when TensorFlow is not available
    from tensorflow.keras.models import Sequential #type: ignore
    from tensorflow.keras.layers import Dense, Dropout #type: ignore
    
    model = Sequential()
    model.add(Dense(units, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units // 2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    return model

def train_neural_network(X_train, y_train, input_dim=None, units=64, dropout_rate=0.2, learning_rate=0.001,
                          batch_size=32, epochs=50, validation_split=0.2):
    """
    Train a neural network model.
    
    Parameters:
    -----------
    X_train : numpy array
        The training features.
    y_train : numpy array
        The training target.
    input_dim : int, optional
        Number of input features. If None, will be inferred from X_train.
    units : int
        Number of units in the hidden layer.
    dropout_rate : float
        Dropout rate for regularization.
    learning_rate : float
        Learning rate for the optimizer.
    batch_size : int
        Batch size for training.
    epochs : int
        Number of epochs to train for.
    validation_split : float
        Fraction of training data to use for validation.
        
    Returns:
    --------
    model : KerasClassifier
        The trained neural network model wrapped as a scikit-learn estimator.
    """
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow is not installed. Please install it to use neural networks.")
        return None
        
    # Import inside function to avoid errors when TensorFlow is not available
    from tensorflow.keras.callbacks import EarlyStopping #type: ignore
    from scikeras.wrappers import KerasClassifier
    
    # Set input dimension if not provided
    if input_dim is None:
        input_dim = X_train.shape[1]
    
    # Create model-building function
    def create_model():
        return create_neural_network(
            input_dim=input_dim,
            units=units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
    
    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Create the KerasClassifier
    model = KerasClassifier(
        model=create_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=validation_split,
        callbacks=[early_stopping]
    )
    
    # Train the model
    logger.info(f"Training neural network with units={units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate the model and return metrics.
    
    Parameters:
    -----------
    model : sklearn model or tensorflow model
        The trained model.
    X_test : numpy array
        The testing features.
    y_test : numpy array
        The testing target.
    feature_names : numpy array, optional
        Names of the features.
        
    Returns:
    --------
    metrics : dict
        A dictionary of evaluation metrics.
    fig : matplotlib.figure.Figure
        The confusion matrix plot figure.
    """
    # Make predictions
    if isinstance(model, tf.keras.Model):
        # Handle TensorFlow models
        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        # Handle sklearn models
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    logger.info(f"Model evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create confusion matrix plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Alzheimer\'s', 'Alzheimer\'s']).plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # If the model has feature importances, plot them
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort importances in descending order
        indices = np.argsort(importances)[::-1]
        top_k = 15  # Show top 15 features
        
        # Create feature importance plot (save to separate file)
        feat_fig = plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(top_k), importances[indices][:top_k], align='center')
        plt.xticks(range(top_k), feature_names[indices][:top_k], rotation=90)
        plt.tight_layout()
        
        # Save feature importance plot
        ensure_directory_exists("reports/figures")
        feat_imp_path = os.path.join("reports", "figures", "feature_importances.png")
        remove_existing_file(feat_imp_path)
        feat_fig.savefig(feat_imp_path)
        plt.close(feat_fig)
    
    # If the model has coefficients, plot them (for logistic regression)
    elif hasattr(model, 'coef_') and feature_names is not None:
        coef = model.coef_[0]
        
        # Sort coefficients by absolute value in descending order
        indices = np.argsort(np.abs(coef))[::-1]
        top_k = 15  # Show top 15 features
        
        # Create coefficients plot (save to separate file)
        coef_fig = plt.figure(figsize=(10, 6))
        plt.title('Feature Coefficients')
        plt.bar(range(top_k), coef[indices][:top_k], align='center')
        plt.xticks(range(top_k), feature_names[indices][:top_k], rotation=90)
        plt.tight_layout()
        
        # Save coefficients plot
        ensure_directory_exists("reports/figures")
        coef_path = os.path.join("reports", "figures", "feature_coefficients.png")
        remove_existing_file(coef_path)
        coef_fig.savefig(coef_path)
        plt.close(coef_fig)
    
    return metrics, fig

def load_best_params(model_type):
    """
    Load the best hyperparameters from hyperparameter tuning.
    
    Parameters:
    -----------
    model_type : str
        The model type to load parameters for.
        
    Returns:
    --------
    params : dict
        The best hyperparameters for the specified model.
    """
    best_params_path = os.path.join("models", "tuned_models", f"{model_type}_best_params.json")
    
    if not os.path.exists(best_params_path):
        logger.warning(f"Best parameters file not found: {best_params_path}")
        return {}
    
    try:
        with open(best_params_path, 'r') as f:
            import json
            params = json.load(f)
        logger.info(f"Loaded best parameters for {model_type} model")
        return params
    except Exception as e:
        logger.error(f"Error loading best parameters: {str(e)}")
        return {}

def main():
    args = parse_args()
    logger.info("Loading preprocessed data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Set up the MLflow experiment with a fixed name for all training
    experiment_name = "trained_models"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: {experiment_name}")
    
    # Get model parameters - either default or best from hyperparameter tuning
    use_best_params = args.use_best_params.lower() == "true"
    best_params = {}
    
    if use_best_params:
        logger.info(f"Using best parameters for {args.model_type} model")
        best_params = load_best_params(args.model_type)
    
    # Model name without run number
    model_name = f"{args.model_type}_model"
    logger.info(f"Training {args.model_type} model")
    
    # Train the specified model
    if args.model_type == "logistic":
        # Get parameters for logistic regression
        alpha = best_params.get('C', 1.0/args.alpha)
        if 'C' in best_params:  # Convert C to alpha
            alpha = 1.0/alpha
        l1_ratio = best_params.get('l1_ratio', args.l1_ratio)
        
        model = train_logistic_regression(X_train, y_train, alpha, l1_ratio)
    elif args.model_type == "rf":
        if use_best_params and best_params:
            model = RandomForestClassifier(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', 10),
                min_samples_split=best_params.get('min_samples_split', 5),
                min_samples_leaf=best_params.get('min_samples_leaf', 2),
                class_weight='balanced',
                random_state=RANDOM_SEED
            )
            model.fit(X_train, y_train)
        else:
            model = train_random_forest(X_train, y_train)
    elif args.model_type == "knn":
        if use_best_params and best_params:
            model = KNeighborsClassifier(
                n_neighbors=best_params.get('n_neighbors', 5),
                weights=best_params.get('weights', 'uniform'),
                p=best_params.get('p', 2),
                leaf_size=best_params.get('leaf_size', 30),
                algorithm='auto'
            )
            model.fit(X_train, y_train)
        else:
            model = train_knn(X_train, y_train)
    elif args.model_type == "svm":
        if use_best_params and best_params:
            model = SVC(
                C=best_params.get('C', 1.0),
                kernel=best_params.get('kernel', 'rbf'),
                gamma=best_params.get('gamma', 'scale'),
                class_weight='balanced',
                probability=True,
                random_state=RANDOM_SEED
            )
            model.fit(X_train, y_train)
        else:
            model = train_svm(X_train, y_train)
    elif args.model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost is not installed. Please install it with 'pip install xgboost'")
            return
            
        if use_best_params and best_params:
            # Validate parameters to ensure they're within valid ranges
            n_estimators = int(best_params.get('n_estimators', 100))
            learning_rate = max(0.001, float(best_params.get('learning_rate', 0.1)))  # Must be positive
            max_depth = max(1, int(best_params.get('max_depth', 3)))  # Must be at least 1
            subsample = min(1.0, max(0.0, float(best_params.get('subsample', 0.8))))  # Must be in [0,1]
            
            logger.info(f"Using validated XGBoost parameters: n_estimators={n_estimators}, "
                        f"learning_rate={learning_rate}, max_depth={max_depth}, subsample={subsample}")
            
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=RANDOM_SEED
            )
            model.fit(X_train, y_train)
        else:
            model = train_xgboost(X_train, y_train)
    elif args.model_type == "nn":
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is not installed. Please install it with 'pip install tensorflow'")
            return
            
        if use_best_params and best_params:
            # Validate parameters to ensure they're within valid ranges
            units = int(best_params.get('units', 64))
            dropout_rate = max(0.0, min(0.9, float(best_params.get('dropout_rate', 0.2))))
            learning_rate = max(0.0001, float(best_params.get('learning_rate', 0.001)))
            batch_size = int(best_params.get('batch_size', 32))
            epochs = int(best_params.get('epochs', 50))
            
            logger.info(f"Using validated Neural Network parameters: units={units}, "
                       f"dropout_rate={dropout_rate}, learning_rate={learning_rate}, "
                       f"batch_size={batch_size}, epochs={epochs}")
            
            model = train_neural_network(
                X_train, y_train,
                units=units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs
            )
        else:
            model = train_neural_network(X_train, y_train)
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return
    
    # Evaluate the model
    logger.info("Evaluating model...")
    metrics, fig = evaluate_model(model, X_test, y_test, feature_names)
    
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Create models directory if it doesn't exist
    ensure_directory_exists(os.path.join("models", "trained_models"))
    
    # Save the model
    model_path = os.path.join("models", "trained_models", f"{model_name}.pkl")
    remove_existing_file(model_path)
    
    # Special handling for TensorFlow/Keras models
    if args.model_type == "nn":
        # Save TensorFlow model in .keras format
        tf_model_path = os.path.join("models", "trained_models", f"{model_name}.keras")
        remove_existing_file(tf_model_path)
        model.save(tf_model_path)
        logger.info(f"Neural network model saved to {tf_model_path}")
    else:
        # Save sklearn models using joblib
        joblib.dump(model, model_path)
    
    # Save the confusion matrix figure, overwrite if exists
    ensure_directory_exists(os.path.join("reports", "figures"))
    fig_path = os.path.join("reports", "figures", f"{model_name}_confusion_matrix.png")
    remove_existing_file(fig_path)
    fig.savefig(fig_path)
    
    # Add timestamp to run name for uniqueness instead of deleting previous runs
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_name = f"{model_name}_{timestamp}"
    
    # Log the model and metrics with MLflow
    with mlflow.start_run(run_name=unique_run_name):
        # Log model parameters
        mlflow.log_param("model_type", args.model_type)
        if args.model_type == "logistic":
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
        
        # Log whether we used best parameters
        mlflow.log_param("use_best_params", use_best_params)
        
        # Log all metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Special handling for logging TensorFlow models to MLflow
        if args.model_type == "nn":
            # Log the TensorFlow model to MLflow
            sample_input = pd.DataFrame(X_test[0:1], columns=feature_names)
            mlflow.tensorflow.log_model(
                model,
                model_name,
                signature=mlflow.models.infer_signature(X_train, y_train),
                input_example=sample_input
            )
        else:
            # Log sklearn models to MLflow
            sample_input = pd.DataFrame(X_test[0:1], columns=feature_names)
            mlflow.sklearn.log_model(
                model,
                model_name,
                signature=mlflow.models.infer_signature(X_train, y_train),
                input_example=sample_input
            )
        
        # Register the model without run number
        registered_model_name = f"alzheimers_{model_name}"
        
        # Get MLflow client for model registration
        client = mlflow.tracking.MlflowClient()
        
        # Check if the registered model already exists
        try:
            client.get_registered_model(registered_model_name)
            logger.info(f"Updating existing registered model: {registered_model_name}")
        except:
            logger.info(f"Creating new registered model: {registered_model_name}")
        
        # Register the model (creates a new version if model exists, or creates model if it doesn't)
        model_details = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
            registered_model_name
        )
        
        # Transition this version to Production (replacing previous production version)
        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_details.version,
            stage="Production"
        )
        
        # Log the model version
        mlflow.log_param("model_version", model_details.version)
        
        # Log the confusion matrix figure
        mlflow.log_artifact(fig_path)
        
        # Explicitly mark run as successful
        mlflow.set_tag("status", "SUCCESS")
        
        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        # Log hyperlink to the MLflow UI
        experiment_id = mlflow.get_experiment_by_name("trained_models").experiment_id
        mlflow_url = f"file://{os.getcwd()}/mlruns/#/experiments/{experiment_id}/runs/{run_id}"
        logger.info(f"MLflow experiment URL: {mlflow_url}")
    
    if args.model_type == "nn":
        logger.info(f"Model training completed. Neural network model saved to {tf_model_path}")
    else:
        logger.info(f"Model training completed. Model saved to {model_path}")
    logger.info(f"Model registered in MLflow as {registered_model_name}")

if __name__ == "__main__":
    main() 