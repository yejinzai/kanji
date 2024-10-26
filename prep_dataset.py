# prepare_dataset.py

import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image
import os


def load_kanji_dataset(csv_file):
    # Load metadata CSV
    df = pd.read_csv(csv_file)

    # Replace missing values or NaNs with empty strings
    df.fillna("", inplace=True)

    # Ensure all columns are of the correct data type
    df["meanings"] = df["meanings"].astype(str)
    df["png_path"] = df["png_path"].astype(str)

    # Create separate rows for each individual meaning in a semicolon-separated string
    new_rows = []
    for _, row in df.iterrows():
        meanings = row["meanings"].split(";")
        #for meaning in meanings:
        #    new_rows.append({
        #        "text": meaning.strip()[0],
        #        "image": row["png_path"]
        #    })
        if meanings[0].strip():
            new_rows.append({
                "text": meanings[0].strip(),
                "image": row["png_path"]
            })
    print(len(new_rows))
    # Create a new DataFrame with the expanded rows
    new_df = pd.DataFrame(new_rows)

    # Prepare the dataset format for Hugging Face's Dataset
    data = {
        "text": new_df["text"].tolist(),
        "image": new_df["image"].tolist()
    }
    dataset = Dataset.from_dict(data)

    # Split into train and validation sets
    dataset = dataset.train_test_split(test_size=0.1)

    return dataset


def save_dataset(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    dataset = load_kanji_dataset("flat_kanji/metadata.csv")
    save_dataset(dataset, "kanji_dataset2")
    print("Dataset prepared and saved to 'kanji_dataset'.")
