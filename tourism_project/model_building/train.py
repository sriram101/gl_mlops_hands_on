# Production script for model training with MLflow experiment tracking

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# Configure MLflow tracking (localhost for GitHub Actions environment)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-prediction-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load train and test data from the Hugging Face data space
Xtrain = pd.read_csv("hf://datasets/sriram-acad/tourism-data/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/sriram-acad/tourism-data/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/sriram-acad/tourism-data/ytrain.csv").values.ravel()
ytest = pd.read_csv("hf://datasets/sriram-acad/tourism-data/ytest.csv").values.ravel()

# Define numeric and categorical feature lists
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# Compute class weight to handle target imbalance
class_weight = sum(ytrain == 0) / sum(ytrain == 1)

# Define preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define the base XGBoost classifier
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.7],
    "xgbclassifier__reg_lambda": [0.5, 1.0],
}

# Build the model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():

    # Hyperparameter tuning with cross-validation
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their scores as nested runs
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log the best parameters in the parent run
    mlflow.log_params(grid_search.best_params_)

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log evaluation metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"],
    })

    print("Train Classification Report:")
    print(classification_report(ytrain, y_pred_train))
    print("Test Classification Report:")
    print(classification_report(ytest, y_pred_test))

    # Save the best model locally
    model_path = "best_tourism_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact: {model_path}")

    # Register the best model on the Hugging Face model hub
    model_repo_id = "sriram-acad/tourism-model"
    model_repo_type = "model"

    try:
        api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
        print(f"Model repo '{model_repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{model_repo_id}'...")
        create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False)
        print(f"Model repo '{model_repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_model.joblib",
        repo_id=model_repo_id,
        repo_type=model_repo_type,
    )
    print("Best model registered on Hugging Face model hub.")
