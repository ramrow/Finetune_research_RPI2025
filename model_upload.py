from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token))
api.upload_folder(
    folder_path="./foamqwen",
    repo_id="finalform/foamqwen-unsloth",
    repo_type="model",
)