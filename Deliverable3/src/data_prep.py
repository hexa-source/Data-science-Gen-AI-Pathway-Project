

"""This module contains data preparation functions for the fake news classification project.
It includes functions to load the dataset, clean the text data."""

import pandas as pd

def load_and_prepare_data(file_path: str):
    """
    Load the dataset from a CSV file, clean the text data, and split it into training and testing sets.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility."""

    df = pd.read_csv(file_path)
    df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[df["content"].str.strip() != ""]
    df = df.drop_duplicates(subset="content").reset_index(drop=True)

    return df





cat_feat = "clean_text"