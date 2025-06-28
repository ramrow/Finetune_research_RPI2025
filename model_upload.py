from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token))
api.upload_folder(
    folder_path="./qwen-foam",
    repo_id="finalform/foamqwen-7B",
    repo_type="model",
)