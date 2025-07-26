from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token))
api.upload_folder(
    folder_path="./factory_qwen_results",
    repo_id="finalform/testfoam",
    repo_type="model",
)