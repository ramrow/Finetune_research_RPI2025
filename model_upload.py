from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token))
api.upload_folder(
    folder_path="bernardFOAM.csv",
    repo_id="finalform/bernardFOAM",
    repo_type="dataset",
)