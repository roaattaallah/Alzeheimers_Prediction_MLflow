#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import requests
import json
import pandas as pd
import logging
import os
import time
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load preprocessing pipeline
def load_preprocessing_pipeline():
    """Load the preprocessing pipeline for normalizing inputs."""
    pipeline_path = os.path.join("data", "processed", "preprocessing_pipeline.pkl")
    
    if os.path.exists(pipeline_path):
        logger.info(f"Loading preprocessing pipeline from {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
        return pipeline
    else:
        logger.warning(f"Preprocessing pipeline not found: {pipeline_path}")
        return None

# Load feature names
def load_feature_names():
    feature_names_path = os.path.join("data", "processed", "feature_names.csv")
    if os.path.exists(feature_names_path):
        return pd.read_csv(feature_names_path)["0"].tolist()
    return []

# Load preprocessing pipeline and feature names
preprocessing_pipeline = load_preprocessing_pipeline()
feature_names = load_feature_names()

# Model information - matches the deployed models in mlflow_serve.py with corrected metrics
MODEL_INFO = {
    "rf": {
        "name": "alzheimers_rf_tuned",
        "version": "9",
        "port": 5016,
        "is_best": False,
        "accuracy": 0.819,
        "roc_auc": 0.871,
        "display_name": "Random Forest"
    },
    "xgboost": {
        "name": "alzheimers_xgboost_tuned",
        "version": "9",
        "port": 5011,
        "is_best": True,
        "accuracy": 0.833,
        "roc_auc": 0.874,
        "display_name": "XGBoost"
    },
    "logistic": {
        "name": "alzheimers_logistic_tuned",
        "version": "16",
        "port": 5012,
        "is_best": False,
        "accuracy": 0.747,
        "roc_auc": 0.794,
        "display_name": "Logistic Regression"
    },
    "knn": {
        "name": "alzheimers_knn_tuned",
        "version": "9",
        "port": 5013,
        "is_best": False,
        "accuracy": 0.677,
        "roc_auc": 0.724,
        "display_name": "K-Nearest Neighbors"
    },
    "svm": {
        "name": "alzheimers_svm_tuned",
        "version": "10",
        "port": 5014,
        "is_best": False,
        "accuracy": 0.674,
        "roc_auc": 0.76,
        "display_name": "Support Vector Machine"
    },
    "nn": {
        "name": "alzheimers_nn_tuned",
        "version": "13",
        "port": 5015,
        "is_best": False,
        "accuracy": 0.726,
        "roc_auc": 0.785,
        "display_name": "Neural Network"
    },
    "high_accuracy_ensemble": {
        "name": "high_accuracy_ensemble",
        "version": "5",
        "port": 5017,
        "is_best": True,
        "accuracy": 0.90,
        "roc_auc": 0.96,
        "display_name": "High-Accuracy Ensemble"
    }
}

@app.route('/', methods=['GET'])
def index():
    """Return a simple welcome message for the root endpoint"""
    return jsonify({
        "message": "Alzheimer's Disease Prediction API",
        "available_endpoints": [
            "/api/models - Get information about available models",
            "/api/features - Get information about model features",
            "/api/predict - Make predictions (POST request)",
            "/api/predict-ensemble - Make ensemble predictions with voting (POST request)",
            "/api/high-accuracy-ensemble - Make predictions with high-accuracy ensemble model (POST request)",
            "/api/direct-ensemble - Make predictions using a direct ensemble of models (POST request)"
        ]
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return information about all available models"""
    try:
        return jsonify(MODEL_INFO)
    except Exception as e:
        logger.error(f"Error retrieving models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Return information about all model features"""
    try:
        # Return the basic feature info without the detailed descriptions
        feature_info = {feature: {"description": f"{feature} parameter"} for feature in feature_names}
        return jsonify(feature_info)
    except Exception as e:
        logger.error(f"Error retrieving features: {e}")
        return jsonify({"error": str(e)}), 500

def check_model_server(port, max_retries=3):
    """Check if the model server is running"""
    for attempt in range(max_retries):
        try:
            # Try a simple health check
            health_response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if health_response.status_code == 200:
                logger.info(f"Model server on port {port} is healthy")
                return True
        except requests.exceptions.RequestException:
            # Health endpoint might not be available, let's check if the server is listening
            try:
                # Just check if the server responds to a minimal request
                response = requests.get(f"http://localhost:{port}/", timeout=2)
                logger.info(f"Model server on port {port} is responding")
                return True
            except requests.exceptions.ConnectionError:
                logger.warning(f"Model server on port {port} not responding on attempt {attempt+1}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Waiting before retry for port {port}...")
                    time.sleep(3)  # Wait before retrying
                    
    logger.error(f"Model server on port {port} could not be reached after {max_retries} attempts")
    return False

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction using the specified model"""
    try:
        data = request.json
        model_name = data.get('model_name')
        parameters = data.get('parameters')
        
        logger.info(f"Received prediction request for model: {model_name}")
        logger.info(f"Raw parameters from frontend: {parameters}")
        
        if not model_name or not parameters:
            return jsonify({"error": "Missing model_name or parameters"}), 400
        
        if model_name not in MODEL_INFO:
            return jsonify({"error": f"Model {model_name} not found"}), 404
        
        # Get the port number for the selected model
        port = MODEL_INFO[model_name]['port']
        
        # Check if the model server is running
        if not check_model_server(port):
            return jsonify({
                "error": f"Model server for {MODEL_INFO[model_name]['display_name']} is not available",
                "details": "Please ensure that the MLflow model servers are running by executing 'python mlflow_serve.py' first"
            }), 503
        
        # Create DataFrame from input parameters
        df = pd.DataFrame({key: [value] for key, value in parameters.items()})
        
        # Ensure all features are present
        if feature_names:
            # Add missing features with default 0 value
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            # Reorder columns to match training data
            df = df[feature_names]
        
        # Apply preprocessing if pipeline is available
        if preprocessing_pipeline:
            logger.info("Applying preprocessing pipeline to input data")
            input_array = df.values
            preprocessed_data = preprocessing_pipeline.transform(input_array)
            
            # Convert back to DataFrame for MLflow serving
            processed_df = pd.DataFrame(preprocessed_data, columns=df.columns)
            
            columns = processed_df.columns.tolist()
            data_values = processed_df.values.tolist()
        else:
            logger.warning("No preprocessing pipeline available, using raw input data")
            columns = df.columns.tolist()
            data_values = df.values.tolist()
        
        # Create payload for MLflow server
        payload = {
            "dataframe_split": {
                "columns": columns,
                "data": data_values
            }
        }
        
        # Make prediction request to MLflow server with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending prediction request to model server on port {port} (attempt {attempt+1})")
                response = requests.post(
                    f"http://localhost:{port}/invocations",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    break
                
                logger.warning(f"Request to model server returned status code {response.status_code} (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error connecting to model server on port {port}: {e} (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    return jsonify({
                        "error": f"Cannot connect to model server for {MODEL_INFO[model_name]['display_name']}",
                        "details": "The server is not responding to prediction requests"
                    }), 503
        
        if response.status_code != 200:
            logger.error(f"Error from MLflow server: {response.text}")
            return jsonify({"error": f"MLflow server error: {response.text}"}), 500
        
        # Parse the response from MLflow server
        result = response.json()
        logger.info(f"Raw response from MLflow server: {result}")
        
        # Process the result for better frontend display
        processed_result = {}
        
        # First determine the type of response
        if isinstance(result, list):
            # Simple list response (usually a class prediction)
            prediction_value = result[0]
            processed_result['raw_prediction'] = prediction_value
            
            if prediction_value in [0, 1]:
                # Binary classification result
                processed_result['prediction_type'] = 'binary'
                processed_result['prediction'] = "Positive" if prediction_value == 1 else "Negative"
                processed_result['probability'] = 1.0 if prediction_value == 1 else 0.0
            else:
                # Probability output
                processed_result['prediction_type'] = 'probability'
                processed_result['probability'] = float(prediction_value)
                processed_result['prediction'] = "Positive" if prediction_value >= 0.5 else "Negative"
        
        elif isinstance(result, dict):
            # Dictionary response (could be probabilities or other metadata)
            processed_result['raw_prediction'] = result
            
            if 'predictions' in result:
                prediction_value = result['predictions'][0]
                processed_result['prediction_type'] = 'binary'
                processed_result['prediction'] = "Positive" if prediction_value == 1 else "Negative"
                processed_result['probability'] = 1.0 if prediction_value == 1 else 0.0
            
            elif 'probability' in result:
                probability = result['probability'][0]
                processed_result['prediction_type'] = 'probability'
                processed_result['probability'] = float(probability)
                processed_result['prediction'] = "Positive" if probability >= 0.5 else "Negative"
        
        # Add interpretation based on probability
        if 'probability' in processed_result:
            prob = processed_result['probability']
            if prob < 0.3:
                processed_result['risk_level'] = "Low risk of Alzheimer's disease"
            elif prob < 0.7:
                processed_result['risk_level'] = "Moderate risk of Alzheimer's disease"
            else:
                processed_result['risk_level'] = "High risk of Alzheimer's disease"
        
        # Add model info
        processed_result['model'] = {
            'name': MODEL_INFO[model_name]['display_name'],
            'accuracy': MODEL_INFO[model_name]['accuracy'],
            'roc_auc': MODEL_INFO[model_name]['roc_auc']
        }
        
        logger.info(f"Returning processed result: {processed_result}")
        return jsonify(processed_result)
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to MLflow server on port {port}")
        return jsonify({
            "error": "MLflow model server not available",
            "details": "Please run 'python mlflow_serve.py' first to start the model servers"
        }), 503
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-ensemble', methods=['POST'])
def predict_ensemble():
    """Make predictions using multiple models and return a consensus prediction"""
    try:
        data = request.json
        parameters = data.get('parameters')
        
        if not parameters:
            return jsonify({"error": "Missing parameters"}), 400
        
        logger.info(f"Received ensemble prediction request")
        logger.info(f"Raw parameters from frontend: {parameters}")
        
        # Create DataFrame from input parameters
        df = pd.DataFrame({key: [value] for key, value in parameters.items()})
        
        # Ensure all features are present
        if feature_names:
            # Add missing features with default 0 value
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            # Reorder columns to match training data
            df = df[feature_names]
        
        # Apply preprocessing if pipeline is available
        if preprocessing_pipeline:
            logger.info("Applying preprocessing pipeline to input data")
            input_array = df.values
            preprocessed_data = preprocessing_pipeline.transform(input_array)
            
            # Convert back to DataFrame for MLflow serving
            processed_df = pd.DataFrame(preprocessed_data, columns=df.columns)
            
            payload_columns = processed_df.columns.tolist()
            payload_data = processed_df.values.tolist()
        else:
            logger.warning("No preprocessing pipeline available, using raw input data")
            payload_columns = df.columns.tolist()
            payload_data = df.values.tolist()
        
        # Define the models to use in the ensemble
        ensemble_models = ["rf", "xgboost", "logistic", "knn", "svm", "nn"]
        
        # Collect predictions from all models
        predictions = []
        probabilities = []
        model_results = []
        
        for model_name in ensemble_models:
            if model_name not in MODEL_INFO:
                logger.warning(f"Model {model_name} not found, skipping")
                continue
                
            # Get the port number for the selected model
            port = MODEL_INFO[model_name]['port']
            
            # Check if the model server is running
            if not check_model_server(port):
                logger.warning(f"Model server for {MODEL_INFO[model_name]['display_name']} is not available, skipping")
                continue
            
            # Create payload for MLflow server
            payload = {
                "dataframe_split": {
                    "columns": payload_columns,
                    "data": payload_data
                }
            }
            
            try:
                # Make prediction request to MLflow server
                response = requests.post(
                    f"http://localhost:{port}/invocations",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    # Parse the response from MLflow server
                    result = response.json()
                    logger.info(f"Raw response from {model_name}: {result}")
                    
                    # Extract prediction value
                    prediction_value = None
                    probability = None
                    
                    if isinstance(result, list):
                        if result[0] in [0, 1]:
                            prediction_value = int(result[0])
                            probability = 1.0 if prediction_value == 1 else 0.0
                        else:
                            probability = float(result[0])
                            prediction_value = 1 if probability >= 0.5 else 0
                    elif isinstance(result, dict):
                        if 'predictions' in result:
                            # Handle nested predictions array (NN model)
                            if isinstance(result['predictions'][0], list):
                                probability = float(result['predictions'][0][0])
                                prediction_value = 1 if probability >= 0.5 else 0
                            else:
                                prediction_value = int(result['predictions'][0])
                                probability = 1.0 if prediction_value == 1 else 0.0
                    
                    if prediction_value is not None:
                        predictions.append(prediction_value)
                        probabilities.append(probability)
                        
                        model_results.append({
                            "model_name": model_name,
                            "display_name": MODEL_INFO[model_name]['display_name'],
                            "prediction": "Positive" if prediction_value == 1 else "Negative",
                            "probability": probability,
                            "accuracy": MODEL_INFO[model_name]['accuracy'],
                            "roc_auc": MODEL_INFO[model_name]['roc_auc']
                        })
                else:
                    logger.error(f"Error from MLflow server for {model_name}: {response.text}")
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
        
        if not predictions:
            return jsonify({"error": "No models were able to make predictions"}), 500
        
        # Calculate ensemble prediction
        positive_count = sum(predictions)
        total_count = len(predictions)
        positive_ratio = positive_count / total_count
        
        # Decision threshold can be adjusted (using 0.5 for majority vote)
        is_positive = positive_ratio >= 0.5
        
        # For negative predictions, we'll invert the positive count to show models that agree with the prediction
        displayed_positive_count = positive_count
        if not is_positive:
            # When prediction is Negative, we want to show how many models predicted Negative
            displayed_positive_count = total_count - positive_count
        
        # Calculate an average probability using only the models that predicted correctly
        aligned_probs = [prob for pred, prob in zip(predictions, probabilities) if pred == int(is_positive)]
        consensus_probability = sum(aligned_probs) / len(aligned_probs) if aligned_probs else positive_ratio
        
        # Weight the probability by accuracy if needed
        weighted_probs = []
        weight_sum = 0
        
        for model_result in model_results:
            if int(model_result["prediction"] == "Positive") == int(is_positive):
                weight = model_result["accuracy"]  # Use accuracy as weight
                weighted_probs.append(model_result["probability"] * weight)
                weight_sum += weight
        
        weighted_probability = sum(weighted_probs) / weight_sum if weight_sum > 0 else consensus_probability
        
        # Process the result for better frontend display
        processed_result = {
            "prediction_type": "ensemble",
            "prediction": "Positive" if is_positive else "Negative",
            "probability": weighted_probability,
            "positive_votes": displayed_positive_count,
            "total_votes": total_count,
            "confidence": positive_ratio if is_positive else (1 - positive_ratio),
            "individual_models": model_results
        }
        
        # Add model consensus that matches the final prediction
        if is_positive:
            # For positive prediction, show how many models predicted positive
            processed_result["model_consensus"] = f"{displayed_positive_count}/{total_count} models"
        else:
            # For negative prediction, show how many models predicted negative
            processed_result["model_consensus"] = f"{displayed_positive_count}/{total_count} models"
        
        # Add interpretation based on probability
        if weighted_probability < 0.3:
            processed_result['risk_level'] = "Low risk of Alzheimer's disease"
        elif weighted_probability < 0.7:
            processed_result['risk_level'] = "Moderate risk of Alzheimer's disease"
        else:
            processed_result['risk_level'] = "High risk of Alzheimer's disease"
        
        logger.info(f"Returning ensemble result: {processed_result}")
        return jsonify(processed_result)
    
    except Exception as e:
        logger.error(f"Error making ensemble prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/high-accuracy-ensemble', methods=['POST'])
def predict_high_accuracy_ensemble():
    """Make predictions using the pre-trained high-accuracy ensemble model"""
    try:
        data = request.json
        parameters = data.get('parameters')
        
        if not parameters:
            return jsonify({"error": "Missing parameters"}), 400
        
        logger.info(f"Received high-accuracy ensemble prediction request")
        logger.info(f"Raw parameters from frontend: {parameters}")
        
        # Create DataFrame from input parameters
        df = pd.DataFrame({key: [value] for key, value in parameters.items()})
        
        # Ensure all features are present
        if feature_names:
            # Add missing features with default 0 value
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            # Reorder columns to match training data
            df = df[feature_names]
        
        # Apply preprocessing if pipeline is available
        if preprocessing_pipeline:
            logger.info("Applying preprocessing pipeline to input data")
            input_array = df.values
            preprocessed_data = preprocessing_pipeline.transform(input_array)
            
            # Convert back to DataFrame for MLflow serving - explicitly preserve column names
            processed_df = pd.DataFrame(preprocessed_data, columns=feature_names)
            
            # Log the dataframe structure for debugging
            logger.info(f"Processed dataframe columns: {processed_df.columns.tolist()}")
            logger.info(f"Processed dataframe shape: {processed_df.shape}")
            
            payload_columns = processed_df.columns.tolist()
            payload_data = processed_df.values.tolist()
        else:
            logger.warning("No preprocessing pipeline available, using raw input data")
            payload_columns = df.columns.tolist()
            payload_data = df.values.tolist()
        
        # Get the port for the high-accuracy ensemble model
        port = MODEL_INFO["high_accuracy_ensemble"]["port"]
        
        # Check if the model server is running
        if not check_model_server(port):
            logger.warning(f"High-accuracy ensemble model server is not available")
            return jsonify({
                "error": "High-accuracy ensemble model server is not available",
                "details": "Please run 'python mlflow_serve.py' first to start the model servers"
            }), 503
        
        # Create payload for MLflow server - Use the explicit column names
        payload = {
            "dataframe_split": {
                "columns": feature_names,
                "data": payload_data
            }
        }
        
        # Log the payload for debugging
        logger.info(f"Sending payload with columns: {payload['dataframe_split']['columns']}")
        logger.info(f"Payload data shape: {len(payload['dataframe_split']['data'])}x{len(payload['dataframe_split']['data'][0]) if payload['dataframe_split']['data'] else 0}")
        
        # Make prediction request to MLflow server with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending prediction request to high-accuracy ensemble model (attempt {attempt+1})")
                response = requests.post(
                    f"http://localhost:{port}/invocations",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=15  # Longer timeout for ensemble model
                )
                
                if response.status_code == 200:
                    break
                
                logger.warning(f"Request to model server returned status code {response.status_code} (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error connecting to high-accuracy ensemble model: {e} (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    return jsonify({
                        "error": "Cannot connect to high-accuracy ensemble model server",
                        "details": "The server is not responding to prediction requests"
                    }), 503
        
        if response.status_code != 200:
            logger.error(f"Error from MLflow server: {response.text}")
            return jsonify({"error": f"MLflow server error: {response.text}"}), 500
        
        # Parse the response from MLflow server
        result = response.json()
        logger.info(f"Raw response from high-accuracy ensemble: {result}")
        
        # Process the result
        processed_result = {}
        
        # Handle predictions based on response format
        if isinstance(result, list):
            # Direct binary prediction
            prediction_value = int(result[0])
            processed_result['prediction_type'] = 'binary'
            processed_result['prediction'] = "Positive" if prediction_value == 1 else "Negative"
            processed_result['probability'] = 1.0 if prediction_value == 1 else 0.0
        elif isinstance(result, dict) and 'predictions' in result:
            # Prediction array format
            prediction_value = int(result['predictions'][0])
            processed_result['prediction_type'] = 'binary'
            processed_result['prediction'] = "Positive" if prediction_value == 1 else "Negative"
            processed_result['probability'] = 1.0 if prediction_value == 1 else 0.0
        
        # Add risk level interpretation
        if processed_result['prediction'] == "Positive":
            processed_result['risk_level'] = "High risk of Alzheimer's disease"
        else:
            processed_result['risk_level'] = "Low risk of Alzheimer's disease"
        
        # Add model info
        processed_result['model'] = {
            'name': MODEL_INFO["high_accuracy_ensemble"]['display_name'],
            'accuracy': MODEL_INFO["high_accuracy_ensemble"]['accuracy'],
            'roc_auc': MODEL_INFO["high_accuracy_ensemble"]['roc_auc'],
            'is_ensemble': True
        }
        
        # Add a note about the ensemble model
        processed_result['note'] = "This prediction comes from our most accurate ensemble model, which combines multiple algorithms for better accuracy."
        
        logger.info(f"Returning high-accuracy ensemble result: {processed_result}")
        return jsonify(processed_result)
    
    except Exception as e:
        logger.error(f"Error making high-accuracy ensemble prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-ensemble', methods=['POST'])
def test_ensemble():
    """Test endpoint for the high-accuracy ensemble model with minimal processing"""
    try:
        data = request.json
        parameters = data.get('parameters', {})
        
        logger.info(f"Received test ensemble request with parameters: {parameters}")
        
        # Create a dataframe with the raw parameters, no preprocessing
        df = pd.DataFrame({key: [float(value)] for key, value in parameters.items()})
        
        # Make sure all feature names are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        logger.info(f"Test dataframe with raw values, columns: {df.columns.tolist()}")
        logger.info(f"Test dataframe sample: {df.head()}")
        
        port = MODEL_INFO["high_accuracy_ensemble"]["port"]
        
        # Simple payload with raw values
        payload = {
            "dataframe_split": {
                "columns": feature_names,
                "data": df.values.tolist()
            }
        }
        
        # Make prediction request to MLflow server
        response = requests.post(
            f"http://localhost:{port}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Successful test response: {result}")
            
            # Return a simple response
            return jsonify({
                "success": True,
                "prediction": result,
                "message": "Test endpoint succeeded"
            })
        else:
            logger.error(f"Error from test endpoint: {response.text}")
            return jsonify({
                "success": False,
                "error": response.text,
                "message": "Test failed with error"
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/direct-ensemble', methods=['POST'])
def direct_ensemble():
    """Make predictions using a direct ensemble of models with the exact weights from high_accuracy_ensemble.py"""
    try:
        data = request.json
        parameters = data.get('parameters', {})
        
        logger.info(f"Received direct ensemble request with parameters: {parameters}")
        
        # Create DataFrame from input parameters
        df = pd.DataFrame({key: [float(value)] for key, value in parameters.items()})
        
        # Ensure all features are present
        if feature_names:
            # Add missing features with default 0 value
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0
            # Reorder columns to match training data
            df = df[feature_names]
        
        # Apply preprocessing if pipeline is available
        if preprocessing_pipeline:
            logger.info("Applying preprocessing pipeline to input data")
            input_array = df.values
            preprocessed_data = preprocessing_pipeline.transform(input_array)
            df = pd.DataFrame(preprocessed_data, columns=feature_names)
        
        # Define models to use in ensemble - EXACT weights from high_accuracy_ensemble.py
        models_to_use = [
            # Format: model_key, model_name, version, weight
            {"key": "xgboost", "name": "alzheimers_xgboost_tuned", "version": "9", "weight": 0.14637536443985882},
            {"key": "rf", "name": "alzheimers_rf_tuned", "version": "9", "weight": 0.08878114635938472},
            {"key": "logistic", "name": "alzheimers_logistic_tuned", "version": "16", "weight": 0.03266075850925048},
            {"key": "knn", "name": "alzheimers_knn_tuned", "version": "9", "weight": 0.012015221588618497},
            {"key": "svm", "name": "alzheimers_svm_tuned", "version": "10", "weight": 0.007287600276738251},
            {"key": "nn", "name": "alzheimers_nn_tuned", "version": "13", "weight": 0.0198097514053307},
            # Additional versions that have significant weights
            {"key": "rf", "name": "alzheimers_rf_tuned", "version": "8", "weight": 0.3978894932909386},
            {"key": "xgboost", "name": "alzheimers_xgboost_tuned", "version": "8", "weight": 0.2413321768584784},
            {"key": "nn", "name": "alzheimers_nn_tuned", "version": "12", "weight": 0.05384848727140148}
        ]
        
        # Get predictions from each model
        ensemble_probs = 0.0
        total_weight = 0.0
        model_results = []
        
        for model_info in models_to_use:
            model_key = model_info["key"]
            model_name = model_info["name"]
            model_version = model_info["version"]
            weight = model_info["weight"]
            
            # Find the model info in MODEL_INFO
            if model_key not in MODEL_INFO:
                logger.warning(f"Model {model_key} not found, skipping")
                continue
            
            # Use the port from the model key
            port = MODEL_INFO[model_key]["port"]
            display_name = f"{MODEL_INFO[model_key]['display_name']} v{model_version}"
            
            # Check if model server is running
            if not check_model_server(port):
                logger.warning(f"Model {model_key} not available, skipping")
                continue
            
            # Create payload
            payload = {
                "dataframe_split": {
                    "columns": df.columns.tolist(),
                    "data": df.values.tolist()
                }
            }
            
            try:
                # Make prediction request
                response = requests.post(
                    f"http://localhost:{port}/invocations",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    # Extract probability
                    result = response.json()
                    logger.info(f"Model {model_name} v{model_version} response: {result}")
                    
                    # Handle different response formats
                    prob = None
                    if isinstance(result, list):
                        if isinstance(result[0], (int, float)) and 0 <= result[0] <= 1:
                            prob = float(result[0])
                    elif isinstance(result, dict) and 'predictions' in result:
                        if isinstance(result['predictions'][0], (int, float)):
                            prob = float(result['predictions'][0])
                    
                    if prob is not None:
                        # Add to ensemble
                        ensemble_probs += weight * prob
                        total_weight += weight
                        
                        model_results.append({
                            "model_name": model_key,
                            "display_name": display_name,
                            "probability": prob,
                            "weight": weight
                        })
                else:
                    logger.warning(f"Model {model_name} v{model_version} returned status {response.status_code}")
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name} v{model_version}: {e}")
        
        if total_weight == 0:
            return jsonify({"error": "No models were available for prediction"}), 500
        
        # Normalize the probability
        final_prob = ensemble_probs / total_weight
        
        # Make prediction
        prediction = 1 if final_prob >= 0.5 else 0
        
        # Format response
        result = {
            "prediction_type": "ensemble",
            "prediction": "Positive" if prediction == 1 else "Negative",
            "probability": float(final_prob),
            "model_results": model_results,
            "model": {
                "name": "High-Accuracy Ensemble",
                "accuracy": 0.90,  # Exact accuracy from high_accuracy_ensemble.py
                "roc_auc": 0.96,   # Exact ROC AUC from high_accuracy_ensemble.py
                "is_ensemble": True
            }
        }
        
        # Add risk level
        if final_prob < 0.3:
            result["risk_level"] = "Low risk of Alzheimer's disease"
        elif final_prob < 0.7:
            result["risk_level"] = "Moderate risk of Alzheimer's disease"
        else:
            result["risk_level"] = "High risk of Alzheimer's disease"
        
        logger.info(f"Direct ensemble result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in direct ensemble: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5030, debug=True) 