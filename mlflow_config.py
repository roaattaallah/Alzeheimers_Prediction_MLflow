import os
import mlflow
from pathlib import Path

db_dir = Path("mlflow_db")
db_dir.mkdir(exist_ok=True)

# tracking URI 
TRACKING_URI = f"sqlite:///{db_dir}/mlflow.db"

def setup_mlflow():

    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    
    if not os.path.exists("mlruns"):
        os.makedirs("mlruns")
        print("Created mlruns directory")

if __name__ == "__main__":
    setup_mlflow()
    print("MLflow configuration complete.")
    print(f" run: mlflow ui --backend-store-uri {TRACKING_URI}")