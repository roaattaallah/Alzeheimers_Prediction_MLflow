# Alzheimer's Disease Prediction Project

This project implements a machine learning pipeline for predicting Alzheimer's disease using various models and ensemble techniques. The project uses MLflow for experiment tracking and model management.

## Project Structure

```
MLFlow_Project/
├── data/
│   ├── Monitoring/           # Monitoring data (Synthetic)
│   └── processed/            # Preprocessed data
    └── data/alzheimers_disease_data.csv  # Original data (raw)
│      
├── models/
│   ├── trained_models/       # Base trained models
│   │   ├── logistic_model.pkl
│   │   ├── rf_model.pkl
│   │   └── .....    
│   ├── tuned_models/         # Hyperparameter-tuned models
│   │   ├── logistic_best_params.json
│   │   ├── rf_best_params.json
│   │   └── ...
│   └── compared_models/      # Best model after comparison
│       ├── best_model.pkl    # The selected best model
│       └── best_model_info.json # Information about the best model
├── mlruns/                   # MLflow tracking directory
├── mlflow_db/                # MLflow database
│   └── mlflow.db             # SQLite database 
├── reports/
│   └── figures/              # Visualizations
│       ├── model_comparison.png
│       ├── feature_coefficients.png
│       └── ...
├── api.py                    # Flask API 
├── compare_models.py         # comparing all model performance
├── hyperparameter_tuning.py  # hyperparameter optimization
├── high_accuracy_ensemble.py # creating an ensemble model
├── mlflow_config.py          # MLflow configuration
├── mlflow_serve.py           # Serving the models via MLflow
├── requirements.txt          # Project dependencies
└── train.py                  # Training  models script


```

This project uses MLflow to register and track different experiments.

## MLflow UI
View experiment results and model registry:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_db/mlflow.db
```
Access the UI at `http://localhost:5000`

## Before starting you should clone the repository and install the neccessary dependancies:

```bash
git clone git@github.com:roaattaallah/Alzeheimers_Prediction_MLflow.git

```
then 
```bash
pip install -r requirements.txt
```



### 1. Data Preprocessing
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
- start: Start all model servers in different ports

### 8. Model Monitoring
Monitor model performance in production:

```bash
python monitor.py
```

Options:
- Monitor all models: `python monitor.py --all`
- Monitor specific model: `python monitor.py --model_name [model_name]`
- Simulate drift: `python monitor.py --simulate_drift`



### Running the frontend to test the models 
The project provides a REST API for making predictions and a frontend built specifically to test all different models with the wanted parameters. In order to run the frontend successfully the following has to be done:  

Start the API server
```bash
python api.py
```
Start the frontend by 
```bash
cd frontend
```
then 
```bash
npm install
```
then 

```bash
npm start 
```

This makes the forntend accessible using the URI : http://localhost:3000  


## Notes
- All models and experiments are tracked in MLflow
- Model performance metrics are logged and can be viewed in the MLflow UI
- The project uses SQLite for MLflow tracking
- Model servers run on different ports (5011-5017)
- The API server runs on port 5000 by default

