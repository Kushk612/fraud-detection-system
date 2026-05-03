import kagglehub
import os
import shutil

DATASET = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"

def download_dataset():
    # Download dataset
    path = kagglehub.dataset_download(DATASET)

    print("Downloaded path:", path)

    # Move file to /data for project consistency
    target_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(target_dir, exist_ok=True)

    # Copy all files
    for file in os.listdir(path):
        src = os.path.join(path, file)
        dst = os.path.join(target_dir, file)
        shutil.copy(src, dst)

    print("Dataset moved to /data folder successfully!")

if __name__ == "__main__":
    download_dataset()