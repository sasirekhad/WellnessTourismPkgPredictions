# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Sasimscct/Tourism-Package-Purchase-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

#Drop the unnamed column
df.drop(columns=['Unnamed: 0'], inplace=True)

#Replace 'Fe Male' with 'Female' in Gender Column
df['Gender'].replace('Fe Male', 'Female',inplace=True)

# Replace the value 'unmarried' with 'single' in Martial Status ciolumn
df['MaritalStatus'].replace('Unmarried', 'Single', inplace=True)

# DON'T label encode 'Type' here — train.py pipeline handles it via OneHotEncoder

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="Sasimscct/Tourism-Package-Purchase-Prediction",
        repo_type="dataset",
    )

print("Files uploaded to Hugging Face successfully.")
