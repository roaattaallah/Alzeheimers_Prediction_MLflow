import os
import mlflow
from pathlib import Path

# Create MLflow database directory if it doesn't exist
db_dir = Path("mlflow_db")
db_dir.mkdir(exist_ok=True)

# Define SQLite tracking URI for MLflow
TRACKING_URI = f"sqlite:///{db_dir}/mlflow.db"

def setup_mlflow():
    """Configure MLflow tracking URI and create necessary directories"""
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    
    if not os.path.exists("mlruns"):
        os.makedirs("mlruns")
        print("Created mlruns directory")

if __name__ == "__main__":
    setup_mlflow()
    print("MLflow configuration complete.")
    print(f" run: mlflow ui --backend-store-uri {TRACKING_URI}")