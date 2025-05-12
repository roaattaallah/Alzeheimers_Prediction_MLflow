import os
import mlflow
from pathlib import Path

# Create a database directory if it doesn't exist
db_dir = Path("mlflow_db")
db_dir.mkdir(exist_ok=True)

# Set the tracking URI to use SQLite database
TRACKING_URI = f"sqlite:///{db_dir}/mlflow.db"

def setup_mlflow():
    """Set up MLflow tracking configuration"""
    # Set the tracking URI
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    
    # Create experiment base directory if it doesn't exist
    if not os.path.exists("mlruns"):
        os.makedirs("mlruns")
        print("Created mlruns directory")

# Set up MLflow if this script is run directly
if __name__ == "__main__":
    setup_mlflow()
    print("MLflow configuration complete.")
    print(f"To view the MLflow UI, run: mlflow ui --backend-store-uri {TRACKING_URI}")