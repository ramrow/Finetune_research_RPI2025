from huggingface_hub import HfApi
import os

token = input()
api = HfApi(token=os.getenv(token))
api.upload_folder(
    folder_path="./llama_foam",
    repo_id="finalform/foam-nuTilda-llama-13B",
    repo_type="model",
)