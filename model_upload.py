from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token))
api.upload_large_folder(
    folder_path="foamqwen/checkpoint-312/adapter_config.json",
    repo_id="finalform/foamQwen2.5-7B-Instruct-similar",
    repo_type="model",
)