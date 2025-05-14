import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import mlflow
import logging
from mlflow_config import setup_mlflow
from imblearn.over_sampling import SMOTE
from collections import Counter

DEFAULT_RANDOM_SEED = 42

def set_random_seeds(seed=DEFAULT_RANDOM_SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for data preprocessing"""
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Test size for train-test split (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility (default: 42)")
    parser.add_argument("--smote_ratio", type=float, default=0.8, 
                        help="Target ratio for minority class after SMOTE (default: 0.8)")
    return parser.parse_args()

def create_visualizations(data, y, output_dir="reports/figures/preprocessing"):
    """Generate and save visualizations for exploratory data analysis"""
    os.makedirs(output_dir, exist_ok=True)
    visualization_paths = {}
    
    # Class distribution plot
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=y, palette=['#4472C4', '#ED7D31'])
    plt.title('Target Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Diagnosis', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    total = y.shape[0]
    for i, p in enumerate(ax.patches):
        percentage = 100 * p.get_height() / total
        ax.annotate(f"{percentage:.1f}%", (p.get_x() + p.get_width()/2., p.get_height()/2),
                     ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    plt.xticks([0, 1], ['Healthy (0)', 'Alzheimer\'s (1)'], fontsize=10)
    
    class_dist_path = os.path.join(output_dir, 'class_distribution.png')
    plt.tight_layout()
    plt.savefig(class_dist_path, dpi=300)
    plt.close()
    visualization_paths['class_distribution'] = class_dist_path
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation = data.corr()
    mask = np.triu(correlation)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, annot=False, fmt='.2f', cbar_kws={'shrink': .7})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    corr_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(corr_path, dpi=300)
    plt.close()
    visualization_paths['correlation_heatmap'] = corr_path
    
    # Feature relevance plot
    full_data = data.copy()
    full_data['Diagnosis'] = y
    feature_corrs = abs(full_data.corr()['Diagnosis']).sort_values(ascending=False)
    feature_corrs = feature_corrs[1:]  
    
    plt.figure(figsize=(14, 14))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_corrs)))
    
    plt.barh(range(len(feature_corrs)), feature_corrs.values, color=colors)
    plt.yticks(range(len(feature_corrs)), feature_corrs.index, fontsize=10)
    

    for i, value in enumerate(feature_corrs.values):
        plt.text(value + 0.01, i, f'{value:.3f}', 
                 va='center', fontsize=9, fontweight='bold')
    
    plt.xlabel('Absolute Correlation with Diagnosis', fontsize=14)
    plt.title('All Features Ranked by Relevance to Alzheimer\'s Diagnosis', fontsize=16, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    feat_relevance_path = os.path.join(output_dir, 'top_features_distribution.png')
    plt.tight_layout()
    plt.savefig(feat_relevance_path, dpi=300)
    plt.close()
    visualization_paths['feature_distributions'] = feat_relevance_path
    
    # Top features distribution by diagnosis
    target_correlations = abs(full_data.corr()['Diagnosis']).sort_values(ascending=False)
    top_features = target_correlations[1:10].index.tolist()
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()
    
    fig.suptitle('Distribution of Top 9 Features by Diagnosis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        
        plot_data = full_data.copy()
        plot_data['Diagnosis Group'] = plot_data['Diagnosis'].map({0: 'Healthy', 1: 'Alzheimer\'s'})
        
        sns.violinplot(x='Diagnosis Group', y=feature, data=plot_data, 
                      palette={'Healthy': '#4472C4', 'Alzheimer\'s': '#ED7D31'},
                      inner='quartile', ax=ax)
        

        means = plot_data.groupby('Diagnosis Group')[feature].mean()
        for j, diagnosis in enumerate(['Healthy', 'Alzheimer\'s']):
            mean_val = means[diagnosis]
            ax.text(j, mean_val, f'Mean: {mean_val:.2f}', 
                   ha='center', va='bottom', fontweight='bold', color='black',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        ax.set_title(feature, fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    violin_path = os.path.join(output_dir, 'feature_boxplots.png')
    plt.savefig(violin_path, dpi=300)
    plt.close()
    visualization_paths['feature_boxplots'] = violin_path
    
    return visualization_paths

def create_imbalance_visualization(original_y, resampled_y, output_dir="reports/figures/preprocessing"):
    """Create visualization comparing class distribution before and after resampling"""
    os.makedirs(output_dir, exist_ok=True)
    
    original_counter = Counter(original_y)
    resampled_counter = Counter(resampled_y)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    labels = ['Healthy (0)', 'Alzheimer\'s (1)']
    sizes = [original_counter[0], original_counter[1]]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4472C4', '#ED7D31'])
    plt.title('Original Class Distribution', fontsize=14, fontweight='bold')
    
 
    plt.subplot(1, 2, 2)
    resampled_sizes = [resampled_counter[0], resampled_counter[1]]
    plt.pie(resampled_sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4472C4', '#ED7D31'])
    plt.title('Resampled Class Distribution', fontsize=14, fontweight='bold')
    

    plt.figtext(0.25, 0.01, f"Original: {sizes[0]} Healthy, {sizes[1]} Alzheimer's", ha='center')
    plt.figtext(0.75, 0.01, f"Resampled: {resampled_sizes[0]} Healthy, {resampled_sizes[1]} Alzheimer's", ha='center')
    
    plt.tight_layout()
    

    resampling_viz_path = os.path.join(output_dir, 'class_balancing_effect.png')
    plt.savefig(resampling_viz_path, dpi=300)
    plt.close()
    
    return resampling_viz_path

def handle_class_imbalance(X, y, sampling_ratio=0.8, random_state=DEFAULT_RANDOM_SEED):
    """Apply SMOTE to handle class imbalance in the training data"""
    counter = Counter(y)
    logger.info(f"Original class distribution: {counter}")
    
    logger.info(f"Applying SMOTE with sampling ratio {sampling_ratio}")
    smote = SMOTE(sampling_strategy=sampling_ratio, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    logger.info(f"Class distribution after SMOTE: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def preprocess_data(test_size, random_state, smote_ratio=0.8):
    """Main preprocessing function that loads, transforms and saves the data"""
    set_random_seeds(random_state)
    
    data_path = os.path.join("data", "alzheimers_disease_data.csv")
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Data shape: {data.shape}")
    
    logger.info(f"First few rows of the dataset:\n{data.head()}")
    missing_values = data.isnull().sum()
    logger.info(f"Missing values per column:\n{missing_values}")
    
    # Remove non-feature columns
    data = data.drop(['PatientID', 'DoctorInCharge'], axis=1)
    
    # Convert integer columns to float
    int_columns = data.select_dtypes(include=['int']).columns
    for col in int_columns:
        data[col] = data[col].astype('float64')
    logger.info(f"Converted {len(int_columns)} integer columns to float64")
    
    # Split features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    logger.info(f"Target class distribution:\n{y.value_counts()}")
    
    # Generate visualizations
    logger.info("Generating data visualizations...")
    figures_dir = os.path.join("reports", "figures", "preprocessing")
    os.makedirs(figures_dir, exist_ok=True)
    visualization_paths = create_visualizations(X, y, figures_dir)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Initial training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    
    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('min_max_scaler', MinMaxScaler()),
        ('standard_scaler', StandardScaler())
    ])
    
    # Apply preprocessing
    X_train_scaled = preprocessing_pipeline.fit_transform(X_train)
    X_test_scaled = preprocessing_pipeline.transform(X_test)
    logger.info("Applied MinMaxScaler and StandardScaler for preprocessing")
    
    # Apply SMOTE for class balancing
    logger.info("Applying SMOTE for class balancing")
    X_train_resampled, y_train_resampled = handle_class_imbalance(
        X_train_scaled, y_train, 
        sampling_ratio=smote_ratio,
        random_state=random_state
    )
    
    # Create visualization of class balance before/after
    imbalance_viz_path = create_imbalance_visualization(y_train, y_train_resampled, figures_dir)
    
    # Save processed data
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train_resampled)
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train_resampled)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test.values)
    pd.Series(X_train.columns).to_csv(os.path.join(processed_dir, 'feature_names.csv'), index=False)
    
    # Save preprocessing pipeline
    import joblib
    joblib.dump(preprocessing_pipeline, os.path.join(processed_dir, 'preprocessing_pipeline.pkl'))
    logger.info("Data preprocessing completed successfully.")
    
    # Log to MLflow
    experiment_name = "preprocessing"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: {experiment_name}")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_name = f"data_preprocessing_{timestamp}"
    
    with mlflow.start_run(run_name=unique_run_name) as run:
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples_before_balancing", X_train.shape[0])
        mlflow.log_param("train_samples_after_balancing", X_train_resampled.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("features", X_train.shape[1])
        mlflow.log_param("preprocessing", "MinMaxScaler + StandardScaler")
        mlflow.log_param("smote_ratio", smote_ratio)
        
        original_class_distribution = y.value_counts(normalize=True).to_dict()
        for class_label, proportion in original_class_distribution.items():
            mlflow.log_metric(f"original_class_{class_label}_proportion", proportion)
        
        final_class_distribution = pd.Series(y_train_resampled).value_counts(normalize=True).to_dict()
        for class_label, proportion in final_class_distribution.items():
            mlflow.log_metric(f"final_class_{class_label}_proportion", proportion)
        
        for vis_name, vis_path in visualization_paths.items():
            mlflow.log_artifact(vis_path)
            logger.info(f"Logged visualization: {os.path.basename(vis_path)}")
        
        # Log imbalance visualization
        mlflow.log_artifact(imbalance_viz_path)
        logger.info(f"Logged imbalance visualization: {os.path.basename(imbalance_viz_path)}")
        
        mlflow.set_tag("status", "SUCCESS")
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        mlflow_url = f"file://{os.getcwd()}/mlruns/#/experiments/{experiment_id}/runs/{run_id}"
        logger.info(f"MLflow experiment URL: {mlflow_url}")

def main():
    """Main function to run the preprocessing pipeline"""
    setup_mlflow()
    
    args = parse_args()
    logger.info("Starting data preprocessing...")
    set_random_seeds(args.random_state)
    logger.info(f"Random seed set to {args.random_state} for reproducibility")
    
    preprocess_data(
        args.test_size, 
        args.random_state, 
        smote_ratio=args.smote_ratio
    )
    logger.info("Data preprocessing completed!")

if __name__ == "__main__":
    main() 