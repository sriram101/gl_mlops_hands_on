# Script to push deployment files to the Hugging Face Space for hosting

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "sriram-acad/Tourism-Package-Prediction"
repo_type = "space"

# Check if the space exists; create it if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating space '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, space_sdk="docker")
    print(f"Space '{repo_id}' created.")

# Upload all deployment files to the Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=".",
)
print("Deployment files uploaded to Hugging Face Space.")
