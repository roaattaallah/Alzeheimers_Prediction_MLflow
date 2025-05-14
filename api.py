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
CORS(app)  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_preprocessing_pipeline():
    """load preprocessing pipeline for input data transformation """
    pipeline_path = os.path.join("data", "processed", "preprocessing_pipeline.pkl")
    
    if os.path.exists(pipeline_path):
        return joblib.load(pipeline_path)
    else:
        return None

def load_feature_names():
    """Load feature names used by the models"""
    feature_names_path = os.path.join("data", "processed", "feature_names.csv")
    if os.path.exists(feature_names_path):
        return pd.read_csv(feature_names_path)["0"].tolist()
    return []

preprocessing_pipeline = load_preprocessing_pipeline()
feature_names = load_feature_names()

# Information about available models and their endpoints
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
    """API root endpoint showing available endpoints"""
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
    """Return information about available models"""
    try:
        return jsonify(MODEL_INFO)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Return information about model features"""
    try:
        feature_info = {feature: {"description": f"{feature} parameter"} for feature in feature_names}
        return jsonify(feature_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def check_model_server(port, max_retries=3):
    """Check if model server is running on specified port"""
    for attempt in range(max_retries):
        try:
            health_response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if health_response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            try:
                response = requests.get(f"http://localhost:{port}/", timeout=2)
                return True
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    time.sleep(3)
    return False

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using a single model"""
    try:
        data = request.json
        model_name = data.get('model_name')
        parameters = data.get('parameters')
        
        if not model_name or not parameters:
            return jsonify({"error": "Missing model_name or parameters"}), 400
        
        if model_name not in MODEL_INFO:
            return jsonify({"error": f"Model {model_name} not found"}), 404
        
        port = MODEL_INFO[model_name]['port']
        
        if not check_model_server(port):
            return jsonify({
                "error": f"Model server for {MODEL_INFO[model_name]['display_name']} is not available",
                "details": "Please ensure that the MLflow model servers are running by executing 'python mlflow_serve.py' first"
            }), 503
        
        # Prepare input data
        df = pd.DataFrame({key: [value] for key, value in parameters.items()})
        
        if feature_names:
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            df = df[feature_names]
        
        # Apply preprocessing if available
        if preprocessing_pipeline:
            input_array = df.values
            preprocessed_data = preprocessing_pipeline.transform(input_array)
            processed_df = pd.DataFrame(preprocessed_data, columns=df.columns)
            columns = processed_df.columns.tolist()
            data_values = processed_df.values.tolist()
        else:
            columns = df.columns.tolist()
            data_values = df.values.tolist()
        
        payload = {
            "dataframe_split": {
                "columns": columns,
                "data": data_values
            }
        }
        
        # Retry logic for model server requests
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"http://localhost:{port}/invocations",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    break
                
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return jsonify({
                        "error": f"Cannot connect to model server for {MODEL_INFO[model_name]['display_name']}",
                        "details": "The server is not responding to prediction requests"
                    }), 503
        
        if response.status_code != 200:
            return jsonify({"error": f"MLflow server error: {response.text}"}), 500
        
        result = response.json()
        processed_result = {}
        
        # Process prediction results
        if isinstance(result, list):
            prediction_value = result[0]
            processed_result['raw_prediction'] = prediction_value
            
            if prediction_value in [0, 1]:
                processed_result['prediction_type'] = 'binary'
                processed_result['prediction'] = "Positive" if prediction_value == 1 else "Negative"
                processed_result['probability'] = 1.0 if prediction_value == 1 else 0.0
            else:
                processed_result['prediction_type'] = 'probability'
                processed_result['probability'] = float(prediction_value)
                processed_result['prediction'] = "Positive" if prediction_value >= 0.5 else "Negative"
        
        elif isinstance(result, dict):
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
        
        # Add risk level based on probability
        if 'probability' in processed_result:
            prob = processed_result['probability']
            if prob < 0.3:
                processed_result['risk_level'] = "Low risk of Alzheimer's disease"
            elif prob < 0.7:
                processed_result['risk_level'] = "Moderate risk of Alzheimer's disease"
            else:
                processed_result['risk_level'] = "High risk of Alzheimer's disease"
        
        processed_result['model'] = {
            'name': MODEL_INFO[model_name]['display_name'],
            'accuracy': MODEL_INFO[model_name]['accuracy'],
            'roc_auc': MODEL_INFO[model_name]['roc_auc']
        }
        
        return jsonify(processed_result)
    
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "MLflow model server not available",
            "details": "Please run 'python mlflow_serve.py' first to start the model servers"
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-ensemble', methods=['POST'])
def predict_ensemble():
    """Make prediction using voting ensemble of all available models"""
    try:
        data = request.json
        parameters = data.get('parameters')
        
        if not parameters:
            return jsonify({"error": "Missing parameters"}), 400
        
        # Prepare input data
        df = pd.DataFrame({key: [value] for key, value in parameters.items()})
        
        if feature_names:
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            df = df[feature_names]
        
        # Apply preprocessing if available
        if preprocessing_pipeline:
            input_array = df.values
            preprocessed_data = preprocessing_pipeline.transform(input_array)
            processed_df = pd.DataFrame(preprocessed_data, columns=df.columns)
            payload_columns = processed_df.columns.tolist()
            payload_data = processed_df.values.tolist()
        else:
            payload_columns = df.columns.tolist()
            payload_data = df.values.tolist()
        
        ensemble_models = ["rf", "xgboost", "logistic", "knn", "svm", "nn"]
        
        predictions = []
        probabilities = []
        model_results = []
        
        # Collect predictions from all available models
        for model_name in ensemble_models:
            if model_name not in MODEL_INFO:
                continue
                
            port = MODEL_INFO[model_name]['port']
            
            if not check_model_server(port):
                continue
            
            payload = {
                "dataframe_split": {
                    "columns": payload_columns,
                    "data": payload_data
                }
            }
            
            try:
                response = requests.post(
                    f"http://localhost:{port}/invocations",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
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
            except Exception:
                pass
        
        if not predictions:
            return jsonify({"error": "No models were able to make predictions"}), 500
        
        # Calculate ensemble prediction by voting
        positive_count = sum(predictions)
        total_count = len(predictions)
        positive_ratio = positive_count / total_count
        
        is_positive = positive_ratio >= 0.5
        
        displayed_positive_count = positive_count
        if not is_positive:
            displayed_positive_count = total_count - positive_count
        
        # Calculate consensus probability
        aligned_probs = [prob for pred, prob in zip(predictions, probabilities) if pred == int(is_positive)]
        consensus_probability = sum(aligned_probs) / len(aligned_probs) if aligned_probs else positive_ratio
        
        # Calculate weighted probability based on model accuracy
        weighted_probs = []
        weight_sum = 0
        
        for model_result in model_results:
            if int(model_result["prediction"] == "Positive") == int(is_positive):
                weight = model_result["accuracy"]
                weighted_probs.append(model_result["probability"] * weight)
                weight_sum += weight
        
        weighted_probability = sum(weighted_probs) / weight_sum if weight_sum > 0 else consensus_probability
        
        processed_result = {
            "prediction_type": "ensemble",
            "prediction": "Positive" if is_positive else "Negative",
            "probability": weighted_probability,
            "positive_votes": displayed_positive_count,
            "total_votes": total_count,
            "confidence": positive_ratio if is_positive else (1 - positive_ratio),
            "individual_models": model_results
        }
        
        processed_result["model_consensus"] = f"{displayed_positive_count}/{total_count} models"
        
        if weighted_probability < 0.3:
            processed_result['risk_level'] = "Low risk of Alzheimer's disease"
        elif weighted_probability < 0.7:
            processed_result['risk_level'] = "Moderate risk of Alzheimer's disease"
        else:
            processed_result['risk_level'] = "High risk of Alzheimer's disease"
        
        return jsonify(processed_result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/high-accuracy-ensemble', methods=['POST'])
def predict_high_accuracy_ensemble():
    """Make prediction using the high-accuracy ensemble model"""
    try:
        data = request.json
        parameters = data.get('parameters')
        
        if not parameters:
            return jsonify({"error": "Missing parameters"}), 400
        
        # Prepare input data
        df = pd.DataFrame({key: [value] for key, value in parameters.items()})
        
        if feature_names:
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            df = df[feature_names]
        
        # Apply preprocessing if available
        if preprocessing_pipeline:
            input_array = df.values
            preprocessed_data = preprocessing_pipeline.transform(input_array)
            processed_df = pd.DataFrame(preprocessed_data, columns=feature_names)
            payload_columns = processed_df.columns.tolist()
            payload_data = processed_df.values.tolist()
        else:
            payload_columns = df.columns.tolist()
            payload_data = df.values.tolist()
        
        port = MODEL_INFO["high_accuracy_ensemble"]["port"]
        
        if not check_model_server(port):
            return jsonify({
                "error": "High-accuracy ensemble model server is not available",
                "details": "Please run 'python mlflow_serve.py' first to start the model servers"
            }), 503
        
        payload = {
            "dataframe_split": {
                "columns": feature_names,
                "data": payload_data
            }
        }
        
        # Retry logic for model server requests
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"http://localhost:{port}/invocations",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=15
                )
                
                if response.status_code == 200:
                    break
                
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return jsonify({
                        "error": "Cannot connect to high-accuracy ensemble model server",
                        "details": "The server is not responding to prediction requests"
                    }), 503
        
        if response.status_code != 200:
            return jsonify({"error": f"MLflow server error: {response.text}"}), 500
        
        result = response.json()
        processed_result = {}
        
        # Process prediction results
        if isinstance(result, list):
            prediction_value = int(result[0])
            processed_result['prediction_type'] = 'binary'
            processed_result['prediction'] = "Positive" if prediction_value == 1 else "Negative"
            processed_result['probability'] = 1.0 if prediction_value == 1 else 0.0
        elif isinstance(result, dict) and 'predictions' in result:
            prediction_value = int(result['predictions'][0])
            processed_result['prediction_type'] = 'binary'
            processed_result['prediction'] = "Positive" if prediction_value == 1 else "Negative"
            processed_result['probability'] = 1.0 if prediction_value == 1 else 0.0
        
        if processed_result['prediction'] == "Positive":
            processed_result['risk_level'] = "High risk of Alzheimer's disease"
        else:
            processed_result['risk_level'] = "Low risk of Alzheimer's disease"
        
        processed_result['model'] = {
            'name': MODEL_INFO["high_accuracy_ensemble"]['display_name'],
            'accuracy': MODEL_INFO["high_accuracy_ensemble"]['accuracy'],
            'roc_auc': MODEL_INFO["high_accuracy_ensemble"]['roc_auc'],
            'is_ensemble': True
        }
        
        processed_result['note'] = "This prediction comes from our most accurate ensemble model, which combines multiple algorithms for better accuracy."
        
        return jsonify(processed_result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-ensemble', methods=['POST'])
def test_ensemble():
    """Test endpoint for ensemble model"""
    try:
        data = request.json
        parameters = data.get('parameters', {})
        
        df = pd.DataFrame({key: [float(value)] for key, value in parameters.items()})
        
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        df = df[feature_names]
        
        port = MODEL_INFO["high_accuracy_ensemble"]["port"]
        
        payload = {
            "dataframe_split": {
                "columns": feature_names,
                "data": df.values.tolist()
            }
        }
        
        response = requests.post(
            f"http://localhost:{port}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            
            return jsonify({
                "success": True,
                "prediction": result,
                "message": "Test endpoint succeeded"
            })
        else:
            return jsonify({
                "success": False,
                "error": response.text,
                "message": "Test failed with error"
            }), response.status_code
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/direct-ensemble', methods=['POST'])
def direct_ensemble():
    """Make prediction using a direct weighted ensemble of models"""
    try:
        data = request.json
        parameters = data.get('parameters', {})
        
        df = pd.DataFrame({key: [float(value)] for key, value in parameters.items()})
        
        if feature_names:
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0
            df = df[feature_names]
        
        if preprocessing_pipeline:
            input_array = df.values
            preprocessed_data = preprocessing_pipeline.transform(input_array)
            df = pd.DataFrame(preprocessed_data, columns=feature_names)
        
        # Models to use in ensemble with their weights
        models_to_use = [
            {"key": "xgboost", "name": "alzheimers_xgboost_tuned", "version": "9", "weight": 0.14637536443985882},
            {"key": "rf", "name": "alzheimers_rf_tuned", "version": "9", "weight": 0.08878114635938472},
            {"key": "logistic", "name": "alzheimers_logistic_tuned", "version": "16", "weight": 0.03266075850925048},
            {"key": "knn", "name": "alzheimers_knn_tuned", "version": "9", "weight": 0.012015221588618497},
            {"key": "svm", "name": "alzheimers_svm_tuned", "version": "10", "weight": 0.007287600276738251},
            {"key": "nn", "name": "alzheimers_nn_tuned", "version": "13", "weight": 0.0198097514053307},
            {"key": "rf", "name": "alzheimers_rf_tuned", "version": "8", "weight": 0.3978894932909386},
            {"key": "xgboost", "name": "alzheimers_xgboost_tuned", "version": "8", "weight": 0.2413321768584784},
            {"key": "nn", "name": "alzheimers_nn_tuned", "version": "12", "weight": 0.05384848727140148}
        ]
        
        ensemble_probs = 0.0
        total_weight = 0.0
        model_results = []
        
        # Collect predictions from all models
        for model_info in models_to_use:
            model_key = model_info["key"]
            model_name = model_info["name"]
            model_version = model_info["version"]
            weight = model_info["weight"]
            
            if model_key not in MODEL_INFO:
                continue
            
            port = MODEL_INFO[model_key]["port"]
            display_name = f"{MODEL_INFO[model_key]['display_name']} v{model_version}"
            
            if not check_model_server(port):
                continue
            
            payload = {
                "dataframe_split": {
                    "columns": df.columns.tolist(),
                    "data": df.values.tolist()
                }
            }
            
            try:
                response = requests.post(
                    f"http://localhost:{port}/invocations",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    prob = None
                    if isinstance(result, list):
                        if isinstance(result[0], (int, float)) and 0 <= result[0] <= 1:
                            prob = float(result[0])
                    elif isinstance(result, dict) and 'predictions' in result:
                        if isinstance(result['predictions'][0], (int, float)):
                            prob = float(result['predictions'][0])
                    
                    if prob is not None:
                        ensemble_probs += weight * prob
                        total_weight += weight
                        
                        model_results.append({
                            "model_name": model_key,
                            "display_name": display_name,
                            "probability": prob,
                            "weight": weight
                        })
            except Exception:
                pass
        
        if total_weight == 0:
            return jsonify({"error": "No models were available for prediction"}), 500
        
        # Calculate final weighted probability
        final_prob = ensemble_probs / total_weight
        prediction = 1 if final_prob >= 0.5 else 0
        
        result = {
            "prediction_type": "ensemble",
            "prediction": "Positive" if prediction == 1 else "Negative",
            "probability": float(final_prob),
            "model_results": model_results,
            "model": {
                "name": "High-Accuracy Ensemble",
                "accuracy": 0.90,
                "roc_auc": 0.96,
                "is_ensemble": True
            }
        }
        
        if final_prob < 0.3:
            result["risk_level"] = "Low risk of Alzheimer's disease"
        elif final_prob < 0.7:
            result["risk_level"] = "Moderate risk of Alzheimer's disease"
        else:
            result["risk_level"] = "High risk of Alzheimer's disease"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5030, debug=True) 