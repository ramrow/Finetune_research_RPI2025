from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token))
api.upload_folder(
    folder_path="./llama-foam",
    repo_id="finalform/foam-nuTilda-sft-llama2-13B",
    repo_type="model",
)