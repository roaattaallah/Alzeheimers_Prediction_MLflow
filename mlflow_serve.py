import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow_config import setup_mlflow
import logging
import subprocess
import time
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start_model_server(model_name, model_version, port):
    """Start MLflow model server for a specific model version on the given port"""
    try:
        # Check if port is already in use
        check_port_cmd = f"lsof -i :{port} | grep LISTEN"
        result = subprocess.run(check_port_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            pid = result.stdout.decode().strip().split()[1]
            logger.info(f"Port {port} is already in use by PID {pid}. Will reuse existing server.")
            print(f"✅ Model {model_name} (v{model_version}) already running on port {port} (PID: {pid})")
            return True
        
        # Start model server in background
        log_file = f"mlflow_model_{model_name}_{port}.log"
        cmd = f"nohup mlflow models serve -m models:/{model_name}/{model_version} -p {port} --host 0.0.0.0 --env-manager=local > {log_file} 2>&1 &"
        
        print(f"Starting model {model_name} (v{model_version}) on port {port}...")
        
        process = subprocess.run(cmd, shell=True)
        
        # Wait for server to start, neural networks and ensembles need more time
        max_attempts = 10
        if "nn_tuned" in model_name or "ensemble" in model_name:
            max_attempts = 15
        elif "xgboost" in model_name:
            max_attempts = 12
            
        for i in range(max_attempts):
            time.sleep(3 * (i + 1))
            
            check_cmd = f"lsof -i :{port} | grep LISTEN"
            result = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                pid = result.stdout.decode().strip().split()[1]
                print(f"✅ Model {model_name} (v{model_version}) started on port {port} (PID: {pid})")
                
                try:
                    health_check = requests.get(f"http://localhost:{port}/health", timeout=10)
                    if health_check.status_code == 200:
                        logger.info(f"Health check successful for model {model_name} on port {port}")
                    else:
                        logger.warning(f"Health check returned status {health_check.status_code} for model {model_name} on port {port}")
                except Exception as e:
                    logger.warning(f"Health check failed for model {model_name} on port {port}: {e}")
                    if "nn_tuned" in model_name or "ensemble" in model_name:
                        logger.info(f"Assuming {model_name} is still initializing (TensorFlow models take longer)")
                
                return True
        
        # Server failed to start
        logger.error(f"Failed to start model server for {model_name} on port {port}")
        
        try:
            with open(log_file, 'r') as f:
                log_contents = f.read()
            logger.error(f"Log contents for {model_name} server:\n{log_contents}")
        except Exception as e:
            logger.error(f"Failed to read log file: {e}")
        
        print(f"❌ Failed to start model server for {model_name} on port {port}")
        return False
    except Exception as e:
        logger.error(f"Error starting model server for {model_name}: {e}")
        print(f"❌ Error starting model server for {model_name}: {e}")
        return False

def cleanup_old_logs():
    """Remove old model server log files"""
    try:
        log_files = [f for f in os.listdir('.') if f.startswith('mlflow_model_') and f.endswith('.log')]
        for log_file in log_files:
            os.remove(log_file)
            print(f"Removed old log file: {log_file}")
    except Exception as e:
        print(f"❌ Error cleaning up log files: {e}")

def main():
    setup_mlflow()
    
    # Clean up old log files
    cleanup_old_logs()
    
    # Define models to serve
    tuned_models = [
        {"name": "alzheimers_xgboost_tuned", "version": "9", "port": 5011},
        {"name": "alzheimers_logistic_tuned", "version": "16", "port": 5012},
        {"name": "alzheimers_knn_tuned", "version": "9", "port": 5013},
        {"name": "alzheimers_svm_tuned", "version": "10", "port": 5014},
        {"name": "alzheimers_nn_tuned", "version": "13", "port": 5015},
        {"name": "alzheimers_rf_tuned", "version": "9", "port": 5016},
        {"name": "high_accuracy_ensemble", "version": "6", "port": 5017}
    ]
    
    # Start all model servers
    print("Starting model servers...")
    for model in tuned_models:
        start_model_server(model["name"], model["version"], model["port"])
    
    # Display summary of deployed models
    print("\n===== DEPLOYED MODELS =====")
    for model in tuned_models:
        print(f"{model['name']} - http://localhost:{model['port']}/invocations")
    
    print("\n⚠️ Models are running in the background")

if __name__ == "__main__":
    main()