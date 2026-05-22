# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Experiment logs to the MLflow tracking server running at local server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLops-Training-Experiment")

# Initializing the Hugging face API client
api = HfApi()

#Reading the train and test data from hugging face repo
Xtrain_path = "hf://datasets/deepacsr/predictive-maintenance/Xtrain.csv"
Xtest_path = "hf://datasets/deepacsr/predictive-maintenance/Xtest.csv"
ytrain_path = "hf://datasets/deepacsr/predictive-maintenance/ytrain.csv"
ytest_path = "hf://datasets/deepacsr/predictive-maintenance/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

#Listign numeric columns

numeric_cols = [
    'Engine rpm', #Engine Speed in Revolutions per minute
    'Lub oil pressure', #Pressure of Lubricating oil in kPa
    'Fuel pressure',     #Pressure at which fuel is supplied to engine in kPa
    'Coolant pressure',   #Pressure of the engine coolan in kPa
    'lub oil temp',  #Temperature of Lubricant oil in degree celsius(°C)
    'Coolant temp'  #Temerature of Engine colant in (°C)
]


# Define the preprocessing steps,Standard scalar used for numerical variables
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_cols)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Define hyperparameter grid
# Paramater Lamda not used to reduce the time for model building
"""
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    #'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}
"""
param_grid = {
    'xgbclassifier__n_estimators': [50,  100],
    'xgbclassifier__max_depth': [2,  4],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    #'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Creating Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1',n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.5

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "deepacsr/predictive-maintenance"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_package_prediction_model_v1.joblib",
        path_in_repo="best_package_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
