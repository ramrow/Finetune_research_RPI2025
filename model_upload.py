from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token))
api.upload_folder(
    folder_path="./llama-foam",
    repo_id="finalform/foam-nuTilda-codellama2-13b",
    repo_type="model",
)