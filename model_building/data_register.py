
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "deepacsr/predictive-maintenance" #Hugging face Space id for the Prediction Maintenance application
repo_type = "dataset" # Since we are uploading the data

# Initialize API client to log into Huggingface
#Reading the secret token for Hugging face from the Git Hub enviromnet
api = HfApi(token=os.getenv("HF_TOKEN_PREDMAINT"))

#Check if the space exists, else create it.
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")


#Uploading the file in to the HF: https://huggingface.co/datasets/deepacsr/predictive-maintenance/
api.upload_folder(
    folder_path="./data",
    repo_id=repo_id,
    repo_type=repo_type,
)
