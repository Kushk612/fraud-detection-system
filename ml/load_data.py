import os
import pandas as pd

def load_dataset():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not files:
        raise FileNotFoundError("No CSV file found in data folder")

    file_path = os.path.join(data_dir, files[0])

    print(f"Loading dataset: {file_path}")

    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully!")
    print(f"Shape: {df.shape}")
    print(f"Fraud ratio: {df['Class'].mean():.6f}")

    return df

if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(df['Class'].value_counts())
    print(df.isnull().sum().sum())