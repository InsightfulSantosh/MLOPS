import re
import pandas as pd
import os
# NLTK for natural language processing
import nltk
from nltk.corpus import stopwords    # For stopwords
from nltk.stem import  WordNetLemmatizer # For stemming and lemmatization

# Downloading NLTK data
nltk.download('stopwords')   # Downloading stopwords data
nltk.download('wordnet')     # Downloading WordNet data for lemmatization



def data_cleaning(text_series):
    """Cleans the text data by removing URLs, emails, numbers, and punctuation."""
    number_pattern = r"(?<=\D)\d+|\d+(?=\D)"  # Removes numbers but keeps letters
    url_pattern = r"https?://\S+|www\.\S+"
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    punctuation_pattern = r"[^\w\s]"

    return (
        text_series.astype(str)  # Ensure text is string
        .str.lower()
        .str.replace(url_pattern, " ", regex=True)
        .str.replace(email_pattern, " ", regex=True)
        .str.replace(number_pattern, " ", regex=True)
        .str.replace(punctuation_pattern, " ", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)  # Normalize spaces
    )

def remove_short_words(text_series, min_length=3):
    """Removes words shorter than `min_length` characters."""
    return text_series.apply(lambda x: " ".join([word for word in x.split() if len(word) >= min_length]))

def lemmatization(text_series):
    """Lemmatizes words using WordNetLemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return text_series.apply(lambda x: " ".join([lemmatizer.lemmatize(word, pos="v") for word in x.split()]))

def remove_stopwords(text_series):
    """Removes stopwords from text."""
    stop_words = frozenset(stopwords.words("english"))  # Faster lookup
    return text_series.apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

def normalize(df):
    """Applies text preprocessing steps."""
    df["content"] = data_cleaning(df["content"])
    df["content"] = remove_short_words(df["content"])
    df["content"] = lemmatization(df["content"])
    df["content"] = remove_stopwords(df["content"])
    return df

def main():
    train_data = pd.read_csv("/Users/santoshkumar/Data_science/Tweet_semtiment/data/raw/train/train.csv")
    test_data = pd.read_csv("/Users/santoshkumar/Data_science/Tweet_semtiment/data/raw/test/test.csv")
    
    # Transform the data
    train_processed_data = normalize(train_data)
    test_processed_data = normalize(test_data)

    # Store the data inside data/processed
    data_path = os.path.join("./data", "interim")
    os.makedirs(data_path, exist_ok=True)

    train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
  

if __name__ == "__main__":
    main()
