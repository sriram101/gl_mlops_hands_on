# Script to load, clean, split, and upload prepared data to Hugging Face

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load the dataset directly from the Hugging Face data space
data_path = "hf://datasets/sriram-acad/tourism-data/tourism.csv"
df = pd.read_csv(data_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Data cleaning: remove unnecessary columns (index column and CustomerID)
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)
print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target distribution:\n{df['ProdTaken'].value_counts()}")

# Split features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Train-test split with stratification to preserve class balance
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train set: {Xtrain.shape[0]} rows | Test set: {Xtest.shape[0]} rows")

# Save the split datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload train and test datasets to Hugging Face
repo_id = "sriram-acad/tourism-data"
repo_type = "dataset"

for file_name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded {file_name} to Hugging Face.")

print("Data preparation complete.")
