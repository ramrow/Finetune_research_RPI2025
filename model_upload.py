from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token))
api.upload_folder(
    folder_path="./qwen-foam",
    repo_id="finalform/foamqwen2.5",
    repo_type="model",
)