import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define BASE_FOLDER globally
BASE_FOLDER = "data"

def make_dir():
    """Creates required directories if they don't exist."""
    sub_dirs = ["raw", "raw/train", "raw/test"]  # Fixed missing comma
    for sub in sub_dirs:
        os.makedirs(os.path.join(BASE_FOLDER, sub), exist_ok=True)

def load_data(url):
    """Loads data from the provided URL."""
    return pd.read_csv(url)

def processing(df):
    """Filters, maps sentiment values, and shuffles data."""
    return (
        df[df["sentiment"].isin(["happiness", "sadness"])]
        .drop(columns=["tweet_id"], errors="ignore")  # Avoid KeyError if column is missing
        .assign(sentiment=lambda x: x["sentiment"].map({"happiness": 1, "sadness": 0}))
        .sample(frac=1, random_state=42)  # Shuffle the data
    )

def save_data(final_df):
    """Splits data into train and test sets and saves them."""
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)

    train_df.to_csv(os.path.join(BASE_FOLDER, "raw/train", "train.csv"), index=False)
    test_df.to_csv(os.path.join(BASE_FOLDER, "raw/test", "test.csv"), index=False)

def main():
    make_dir()
    url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"

    df = load_data(url)
    final_df = processing(df)
    save_data(final_df)

if __name__ == "__main__":
    main()
