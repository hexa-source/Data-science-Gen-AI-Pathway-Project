"""This module contains functionality for splitting data into training and testing sets."""


from sklearn.model_selection import train_test_split

from Deliverable3.src.preprocessor import preprocessor


def split(file_path: str, test_size: float = 0.25, random_state: int = 42):

    df = preprocessor(file_path)
    X = df[["clean_text"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size,
                                                        random_state = random_state,
                                                        )
    return X_train, X_test, y_train, y_test