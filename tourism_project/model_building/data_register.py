# Script to register the raw dataset on the Hugging Face dataset space

from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "sriram-acad/tourism-data"
repo_type = "dataset"

# Check if the dataset repository exists; create it if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Upload the raw dataset to Hugging Face
api.upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Raw dataset uploaded to Hugging Face.")
