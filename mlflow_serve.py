#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from mlflow_config import setup_mlflow
import logging
import subprocess
import signal
import time
import sys
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="MLflow Model Serving")
    parser.add_argument("--action", type=str, choices=["start", "stop", "restart", "status"], default="start")
    return parser.parse_args()

def start_model_server(model_name, model_version, port):
    """Start a model server for a specific model version"""
    try:
        # Check if the port is already in use
        check_port_cmd = f"lsof -i :{port} | grep LISTEN"
        result = subprocess.run(check_port_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            pid = result.stdout.decode().strip().split()[1]
            logger.info(f"Port {port} is already in use by PID {pid}. Will reuse existing server.")
            print(f"✅ Model {model_name} (v{model_version}) already running on port {port} (PID: {pid})")
            return True
        
        # Create the command - redirect output to a log file for debugging
        log_file = f"mlflow_model_{model_name}_{port}.log"
        cmd = f"nohup mlflow models serve -m models:/{model_name}/{model_version} -p {port} --host 0.0.0.0 --no-conda > {log_file} 2>&1 &"
        
        print(f"Starting model {model_name} (v{model_version}) on port {port}...")
        
        # Execute the command using shell
        process = subprocess.run(cmd, shell=True)
        
        # Wait to verify server started
        max_attempts = 10 if "nn_tuned" in model_name or "ensemble" in model_name else 5  # More attempts for TF models
        for i in range(max_attempts):  # Try up to max_attempts times with increasing delays
            time.sleep(3 * (i + 1))  # Wait longer each time: 3s, 6s, 9s, etc.
            
            # Check if server is listening on port
            check_cmd = f"lsof -i :{port} | grep LISTEN"
            result = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                pid = result.stdout.decode().strip().split()[1]
                print(f"✅ Model {model_name} (v{model_version}) started on port {port} (PID: {pid})")
                
                # Verify the server is responding to requests
                try:
                    health_check = requests.get(f"http://localhost:{port}/health", timeout=10)  # Increase timeout from 2 to 10
                    if health_check.status_code == 200:
                        logger.info(f"Health check successful for model {model_name} on port {port}")
                    else:
                        logger.warning(f"Health check returned status {health_check.status_code} for model {model_name} on port {port}")
                except Exception as e:
                    logger.warning(f"Health check failed for model {model_name} on port {port}: {e}")
                    # For TensorFlow models or ensemble, continue even if health check fails
                    if "nn_tuned" in model_name or "ensemble" in model_name:
                        logger.info(f"Assuming {model_name} is still initializing (TensorFlow models take longer)")
                
                return True
        
        # If we get here, the server didn't start properly
        logger.error(f"Failed to start model server for {model_name} on port {port}")
        
        # Print the log file contents for debugging
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

def stop_model_server(port):
    """Stop a model server running on a specific port"""
    try:
        # Find process using the port
        check_cmd = f"lsof -i :{port} | grep LISTEN"
        result = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            # Extract PID
            pid = result.stdout.decode().strip().split()[1]
            
            # Kill the process
            kill_cmd = f"kill -9 {pid}"
            subprocess.run(kill_cmd, shell=True)
            print(f"✅ Stopped server on port {port} (PID: {pid})")
            return True
        else:
            print(f"No server found on port {port}")
            return True
    except Exception as e:
        print(f"❌ Error stopping server on port {port}: {e}")
        return False

def stop_all_servers():
    """Stop all running MLflow model servers"""
    try:
        # Find all MLflow model servers using two different methods to be thorough
        stopped_servers = 0
        
        # Method 1: Find by process name
        cmd1 = "ps aux | grep 'mlflow models serve' | grep -v grep | awk '{print $2}'"
        result1 = subprocess.run(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result1.returncode == 0 and result1.stdout:
            pids = result1.stdout.decode().strip().split('\n')
            for pid in pids:
                if pid:
                    kill_cmd = f"kill -9 {pid}"
                    subprocess.run(kill_cmd, shell=True)
                    print(f"✅ Stopped MLflow model server with PID: {pid}")
                    stopped_servers += 1
        
        # Method 2: Find by uvicorn servers running on our specific ports
        ports = [5011, 5012, 5013, 5014, 5015, 5016]  # The ports used by our models
        for port in ports:
            check_cmd = f"lsof -i :{port} | grep LISTEN | awk '{{print $2}}'"
            result = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0 and result.stdout:
                pids = result.stdout.decode().strip().split('\n')
                for pid in pids:
                    if pid:
                        kill_cmd = f"kill -9 {pid}"
                        subprocess.run(kill_cmd, shell=True)
                        print(f"✅ Stopped server on port {port} (PID: {pid})")
                        stopped_servers += 1
        
        if stopped_servers > 0:
            print(f"Stopped {stopped_servers} model servers")
        else:
            print("No active model servers found to stop")
            
        # Wait a moment to ensure processes are terminated
        time.sleep(2)
        
        # Verify no servers are still running
        verify_cmd = "ps aux | grep 'mlflow models serve\\|uvicorn --host 0.0.0.0 --port' | grep -v grep"
        verify_result = subprocess.run(verify_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if verify_result.returncode == 0 and verify_result.stdout:
            remaining = verify_result.stdout.decode().strip().split('\n')
            if len(remaining) > 0:
                print(f"Warning: {len(remaining)} model server processes may still be running:")
                for proc in remaining:
                    print(f"  {proc.strip()}")
                print("You may need to manually terminate these processes")
        
        return True
    except Exception as e:
        print(f"❌ Error stopping servers: {e}")
        return False

def show_status():
    """Show status of running model servers"""
    try:
        # Find all MLflow model servers
        cmd = "ps aux | grep 'mlflow models serve' | grep -v grep"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0 and result.stdout:
            output = result.stdout.decode().strip()
            print("===== RUNNING MLFLOW MODEL SERVERS =====")
            print(output)
            print("=========================================")
        else:
            print("No MLflow model servers are currently running")
    except Exception as e:
        print(f"❌ Error checking server status: {e}")

def cleanup_old_logs():
    """Clean up any old log files"""
    try:
        log_files = [f for f in os.listdir('.') if f.startswith('mlflow_model_') and f.endswith('.log')]
        for log_file in log_files:
            os.remove(log_file)
            print(f"Removed old log file: {log_file}")
    except Exception as e:
        print(f"❌ Error cleaning up log files: {e}")

def main():
    setup_mlflow()
    args = parse_args()
    
    # Clean up any old log files
    cleanup_old_logs()
    
    # Define the specific tuned models to deploy
    tuned_models = [
        
        {"name": "alzheimers_xgboost_tuned", "version": "9", "port": 5011},
        {"name": "alzheimers_logistic_tuned", "version": "16", "port": 5012},
        {"name": "alzheimers_knn_tuned", "version": "9", "port": 5013},
        {"name": "alzheimers_svm_tuned", "version": "10", "port": 5014},
        {"name": "alzheimers_nn_tuned", "version": "13", "port": 5015},
        {"name": "alzheimers_rf_tuned", "version": "9", "port": 5016},
        {"name": "high_accuracy_ensemble", "version": "6", "port": 5017}
        
    ]
    
    if args.action == "status":
        show_status()
        return
    
    elif args.action == "stop":
        stop_all_servers()
        return
    
    elif args.action == "restart":
        stop_all_servers()
        time.sleep(2)
        # Continue to start the models below
    
    # Start the tuned models
    print("Starting model servers...")
    for model in tuned_models:
        start_model_server(model["name"], model["version"], model["port"])
    
    # Print summary
    print("\n===== DEPLOYED MODELS =====")
    for model in tuned_models:
        print(f"{model['name']} - http://localhost:{model['port']}/invocations")
    
    print("\n⚠️ Models are running in the background")
    print("To stop all models: python mlflow_serve.py --action stop")
    print("To check status: python mlflow_serve.py --action status")

if __name__ == "__main__":
    main()