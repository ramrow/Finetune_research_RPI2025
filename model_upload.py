from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token), timeout=120.0)
api.upload_folder(
    folder_path="foamqwen/checkpoint-312",
    repo_id="finalform/foamQwen2.5-7B-Instruct-similar",
    repo_type="model",
)