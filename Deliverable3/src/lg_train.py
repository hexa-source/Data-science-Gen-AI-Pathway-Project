

"""This module contain functionaly for training the model using 
logistic regression classification model to classify whter the news is fake or real"""


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import classification_report, confusion_matrix

# import local modules
# import vectorizer modeule will automaticall pull train_test_split -> preprocessor -> data_prep
from Deliverable3.src.train_test_split import split
from Deliverable3.src.data_prep import cat_feat


def train(file_path: str):
    X_train, X_test, y_train, y_test = split(file_path)
    
    vectorizer = ColumnTransformer(transformers = [
        ("text", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5, max_df=0.8),
         cat_feat)])
    pipeline = Pipeline(steps = [
        ("transformer", vectorizer),
        ("classifier", LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    tr_pred = pipeline.predict(X_train)
    tr_report = classification_report(y_train, tr_pred)
    print(f"Train Classification report::: {tr_report}")
    tr_confusion = confusion_matrix(y_train, tr_pred)


    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    report = classification_report(y_test, y_pred)
    print(f"Test Classification report::: {report}")
    confusion = confusion_matrix(y_test, y_pred)
    return {
        "Train report": tr_report,
        "Train confusion matix": tr_confusion,
        "Test Report": report,
        "Test confusion": confusion}





if __name__ == "__main__":
    # Load and label
    train("data/fake_or_real_news.csv")