# Alzheimer's Disease Prediction Project

This project implements a machine learning pipeline for predicting Alzheimer's disease using various models and ensemble techniques. The project uses MLflow for experiment tracking and model management.

## Project Structure

```
MLFlow_Project/
├── data/
│   ├── raw/           # Original data
│   └── processed/     # Preprocessed data
├── models/
│   ├── trained_models/    # Base trained models
│   ├── tuned_models/      # Models after hyperparameter tuning
│   ├── ensembles/         # Ensemble models
│   └── compared_models/   # Best performing models
├── monitoring/        # Model monitoring data and visualizations
├── reports/          # Generated reports and visualizations
├── mlruns/           # MLflow experiment tracking
└── mlflow_db/        # MLflow SQLite database
```

## Prerequisites

- Python 3.8+
- Required packages (install using `pip install -r requirements.txt`):
  - MLflow
  - Scikit-learn
  - XGBoost
  - TensorFlow
  - Flask
  - Other dependencies listed in requirements.txt

## Project Flow

### 1. Data Preparation
Place your raw data in the `data/raw/` directory. The data should be in a format compatible with the preprocessing script.

### 2. Data Preprocessing
Run the preprocessing script to prepare the data for model training:

```bash
python preprocess.py
```

This script will:
- Handle class imbalance using SMOTE
- Split data into training and testing sets
- Generate preprocessing visualizations
- Save processed data to `data/processed/`
- Create initial reports in `reports/figures/preprocessing/`

### 3. Model Training and Comparison
Run the model comparison script to train and evaluate different models:

```bash
python compare_models.py
```

This will:
- Train base models (Logistic Regression, Random Forest, KNN, SVM, XGBoost, Neural Network)
- Perform hyperparameter tuning for each model
- Generate comparison visualizations
- Save the best model
- Log results to MLflow

### 4. Hyperparameter Tuning
For detailed hyperparameter tuning of specific models:

```bash
python hyperparameter_tuning.py --model_type [model_type]
```

Available model types:
- logistic
- rf (Random Forest)
- knn
- svm
- xgboost
- nn (Neural Network)

### 5. Model Training with Best Parameters
Train models using the best parameters found during tuning:

```bash
python train.py --model_type [model_type] --use_best_params True
```

### 6. Create High-Accuracy Ensemble
Create an ensemble of the best-performing models:

```bash
python high_accuracy_ensemble.py
```

This will:
- Load the best performing models
- Create a weighted ensemble
- Optimize ensemble thresholds
- Register the ensemble in MLflow

### 7. Model Serving
Start the model servers:

```bash
python mlflow_serve.py --action start
```

Available actions:
- start: Start all model servers
- stop: Stop all model servers
- restart: Restart all model servers
- status: Check server status

### 8. Model Monitoring
Monitor model performance in production:

```bash
python monitor.py
```

Options:
- Monitor all models: `python monitor.py --all`
- Monitor specific model: `python monitor.py --model_name [model_name]`
- Simulate drift: `python monitor.py --simulate_drift`

### 9. API Usage
The project provides a REST API for making predictions. Start the API server:

```bash
python api.py
```

Available endpoints:
- `/api/predict`: Make predictions using individual models
- `/api/predict-ensemble`: Make predictions using the ensemble
- `/api/high-accuracy-ensemble`: Make predictions using the high-accuracy ensemble
- `/api/direct-ensemble`: Make predictions using the direct ensemble

## MLflow UI
View experiment results and model registry:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_db/mlflow.db
```

Access the UI at `http://localhost:5000`

## Output Directories

### Models
- `models/trained_models/`: Base trained models
- `models/tuned_models/`: Models after hyperparameter tuning
- `models/ensembles/`: Ensemble models
- `models/compared_models/`: Best performing models

### Reports
- `reports/figures/`: Generated visualizations
- `reports/preprocessing/`: Preprocessing reports
- `reports/model_comparison/`: Model comparison reports

### Monitoring
- `monitoring/plots/`: Performance monitoring visualizations
- `monitoring/logs/`: Monitoring logs
- `monitoring/summary/`: Monitoring summaries

## Notes
- All models and experiments are tracked in MLflow
- Model performance metrics are logged and can be viewed in the MLflow UI
- The project uses SQLite for MLflow tracking
- Model servers run on different ports (5011-5017)
- The API server runs on port 5000 by default

## Troubleshooting
- If model servers fail to start, check the log files in the project root
- For MLflow issues, verify the SQLite database in `mlflow_db/`
- For API issues, check the Flask server logs
- For monitoring issues, check the logs in `monitoring/logs/` 