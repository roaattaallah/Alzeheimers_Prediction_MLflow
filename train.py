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

from mlflow_config import setup_mlflow

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

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
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def remove_existing_file(file_path):
    
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Removed existing file: {file_path}")

def load_data():
    
    processed_dir = os.path.join("data", "processed")
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    feature_names = pd.read_csv(os.path.join(processed_dir, 'feature_names.csv'))['0'].values
    
    return X_train, X_test, y_train, y_test, feature_names

def train_logistic_regression(X_train, y_train, alpha, l1_ratio):
   
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

    logger.info("Training K-Nearest Neighbors Classifier")
    
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2  
    )
    
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    
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
    
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow is not installed. Please install it to use neural networks.")
        return None
        
    # i will import inside function to avoid errors when TensorFlow is not available
    from tensorflow.keras.models import Sequential #type: ignore
    from tensorflow.keras.layers import Dense, Dropout #type: ignore
    
    model = Sequential()
    model.add(Dense(units, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units // 2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    return model

def train_neural_network(X_train, y_train, input_dim=None, units=64, dropout_rate=0.2, learning_rate=0.001,
                          batch_size=32, epochs=50, validation_split=0.2):
   
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow is not installed. Please install it to use neural networks.")
        return None
        
    # iam importing  inside function to avoid errors when TensorFlow is not available
    from tensorflow.keras.callbacks import EarlyStopping #type: ignore
    from scikeras.wrappers import KerasClassifier
    
    if input_dim is None:
        input_dim = X_train.shape[1]
    
    def create_model():
        return create_neural_network(
            input_dim=input_dim,
            units=units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model = KerasClassifier(
        model=create_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=validation_split,
        callbacks=[early_stopping]
    )
    
    logger.info(f"Training neural network with units={units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, feature_names=None):
    
    # Make predictions
    if isinstance(model, tf.keras.Model):
        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
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
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Create confusion matrix plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Alzheimer\'s', 'Alzheimer\'s']).plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        # feature importances
        importances = model.feature_importances_
        
        indices = np.argsort(importances)[::-1]
        top_k = 15 
        
        # Create feature importance plot 
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
    
    elif hasattr(model, 'coef_') and feature_names is not None:
        coef = model.coef_[0]
        
        indices = np.argsort(np.abs(coef))[::-1]
        top_k = 15 
        
        coef_fig = plt.figure(figsize=(10, 6))
        plt.title('Feature Coefficients')
        plt.bar(range(top_k), coef[indices][:top_k], align='center')
        plt.xticks(range(top_k), feature_names[indices][:top_k], rotation=90)
        plt.tight_layout()
        
        ensure_directory_exists("reports/figures")
        coef_path = os.path.join("reports", "figures", "feature_coefficients.png")
        remove_existing_file(coef_path)
        coef_fig.savefig(coef_path)
        plt.close(coef_fig)
    
    return metrics, fig

def load_best_params(model_type):
    
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
    
    setup_mlflow()
    
    logger.info("Loading preprocessed data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    mlflow.set_experiment("trained_models")
    logger.info(f"MLflow experiment set to: trained_models")
    
    if args.model_type == "nn":
        logger.info("Training nn model")
        model = train_neural_network(X_train, y_train)
        
        logger.info("Evaluating model...")
        metrics, fig = evaluate_model(model, X_test, y_test, feature_names)
        
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        tf_model_path = os.path.join("models", "trained_models", "nn_model.keras")
        
        keras_model = model.model_
        keras_model.save(tf_model_path)
        
        #importing from inside the function to avoid errors when TensorFlow is not available
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_run_name = f"nn_model_{timestamp}"
        
        
        with mlflow.start_run(run_name=unique_run_name):
            mlflow.log_param("model_type", "nn")
            mlflow.log_param("units", 64)
            mlflow.log_param("dropout_rate", 0.2)
            mlflow.log_param("learning_rate", 0.001)
            
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            sample_input = pd.DataFrame(X_test[0:1], columns=feature_names)
            mlflow.tensorflow.log_model(
                keras_model,
                "nn_model",
                signature=mlflow.models.infer_signature(X_train, y_train),
                input_example=sample_input
            )
            
            registered_model_name = "alzheimers_nn_model"
            
            client = mlflow.tracking.MlflowClient()
            
            try:
                client.get_registered_model(registered_model_name)
                logger.info(f"Updating existing registered model: {registered_model_name}")
            except:
                logger.info(f"Creating new registered model: {registered_model_name}")
            
            model_details = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/nn_model",
                registered_model_name
            )
            
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_details.version,
                stage="Production"
            )
            
            mlflow.log_param("model_version", model_details.version)
            
            fig_path = os.path.join("reports", "figures", "nn_model_confusion_matrix.png")
            remove_existing_file(fig_path)
            fig.savefig(fig_path)
            mlflow.log_artifact(fig_path)
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            
            experiment_id = mlflow.get_experiment_by_name("trained_models").experiment_id
            mlflow_url = f"file://{os.getcwd()}/mlruns/#/experiments/{experiment_id}/runs/{run_id}"
            logger.info(f"MLflow experiment URL: {mlflow_url}")
        
        logger.info(f"Neural network model saved to {tf_model_path}")
        logger.info(f"Model registered in MLflow as {registered_model_name}")
        return
    
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
        alpha = best_params.get('C', 1.0/args.alpha)
        if 'C' in best_params:  
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
    
    ensure_directory_exists(os.path.join("models", "trained_models"))
    
    model_path = os.path.join("models", "trained_models", f"{model_name}.pkl")
    remove_existing_file(model_path)
    
    if args.model_type == "nn":
        tf_model_path = os.path.join("models", "trained_models", f"{model_name}.keras")
        remove_existing_file(tf_model_path)
        model.save(tf_model_path)
        logger.info(f"Neural network model saved to {tf_model_path}")
    else:
        joblib.dump(model, model_path)
    
    ensure_directory_exists(os.path.join("reports", "figures"))
    fig_path = os.path.join("reports", "figures", f"{model_name}_confusion_matrix.png")
    remove_existing_file(fig_path)
    fig.savefig(fig_path)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_name = f"{model_name}_{timestamp}"
    
    with mlflow.start_run(run_name=unique_run_name):
        # Log model parameters
        mlflow.log_param("model_type", args.model_type)
        if args.model_type == "logistic":
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
        
        mlflow.log_param("use_best_params", use_best_params)
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        if args.model_type == "nn":
            sample_input = pd.DataFrame(X_test[0:1], columns=feature_names)
            mlflow.tensorflow.log_model(
                model,
                model_name,
                signature=mlflow.models.infer_signature(X_train, y_train),
                input_example=sample_input
            )
        else:
            sample_input = pd.DataFrame(X_test[0:1], columns=feature_names)
            mlflow.sklearn.log_model(
                model,
                model_name,
                signature=mlflow.models.infer_signature(X_train, y_train),
                input_example=sample_input
            )
        
        registered_model_name = f"alzheimers_{model_name}"
        
        client = mlflow.tracking.MlflowClient()
        
        try:
            client.get_registered_model(registered_model_name)
            logger.info(f"Updating existing registered model: {registered_model_name}")
        except:
            logger.info(f"Creating new registered model: {registered_model_name}")
        
        model_details = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
            registered_model_name
        )
        
        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_details.version,
            stage="Production"
        )
        
        mlflow.log_param("model_version", model_details.version)
        
        mlflow.log_artifact(fig_path)
        
        mlflow.set_tag("status", "SUCCESS")
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
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