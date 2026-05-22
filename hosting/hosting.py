
from huggingface_hub import HfApi
import os

#Added for force build deployment after the build 
with open("./deployment/version.txt", "w") as f:
    f.write(str(datetime.now()))

api = HfApi(token=os.getenv("HF_TOKEN_PREDMAINT"))
api.upload_folder(
    folder_path="./deployment",     # the local folder containing your files
    repo_id="deepacsr/predictive-maintenance",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
