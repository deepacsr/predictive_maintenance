# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for imputation
from sklearn.impute import SimpleImputer

# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# LOAD DATASET FROM HUGGING FACE
api = HfApi(token=os.getenv("HF_TOKEN_PREDMAINT"))
DATASET_PATH = "hf://datasets/deepacsr/predictive-maintenance/engine_data"

# Read the data in to panda frame
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Check to remove any columns wihout labels
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# There are no redundant columns in our dataset to be removed

# Define target variable
target_col = 'Engine Condition'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split using standard ratio 80:20
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=56
)

# For any missing values let us impute the values
#IMPUTATION DONE AFTER TRAIN/TEST SPLIT TO AVOID LEAKAGE

#listing all the independant varialbes which are numeric in nature in this case
numeric_cols = [
    'Engine rpm', #Engine Speed in Revolutions per minute
    'Lub oil pressure', #Pressure of Lubricating oil in kPa
    'Fuel pressure',     #Pressure at which fuel is supplied to engine in kPa
    'Coolant pressure',   #Pressure of the engine coolan in kPa
    'lub oil temp',  #Temperature of Lubricant oil in degree celsius(°C)
    'Coolant temp'  #Temerature of Engine colant in (°C)
]

# Imputation done based on column type
# For Numeric columns median value is taken
num_imputer = SimpleImputer(strategy="median")
Xtrain[numeric_cols] = num_imputer.fit_transform(Xtrain[numeric_cols])

#PLEASE NOTE: There are no missing values in our dataset, but imputation done considering the
#real word scenario where data will get ingested which may have missign values

# For Test data only transformation is done and NOT FIT_TRANSFORM
Xtest[numeric_cols] = num_imputer.transform(Xtest[numeric_cols])

#Save the Train , Test data into .csv files
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Uploade the Train,Test csv files to the Huggign face
# https://huggingface.co/datasets/deepacsr/predictive-maintenance/
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="deepacsr/predictive-maintenance",
        repo_type="dataset",
    )
