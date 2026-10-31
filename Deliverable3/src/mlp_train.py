"""This module contain functionaly for training the model usong 
Multilayer perceptron neural net model to classify whter the news is fake or real"""

import argparse

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import classification_report, confusion_matrix

# import local modules
# import vectorizer modeule will automaticall pull train_test_split -> preprocessor -> data_prep
from Deliverable3.src.train_test_split import split
from Deliverable3.src.data_prep import cat_feat
from Deliverable3.src.utils import save_artifact



def train(file_path: str):
    X_train, X_test, y_train, y_test = split(file_path)
    
    vectorizer = ColumnTransformer(transformers = [
        ("text", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5, max_df=0.8),
         cat_feat)])
    pipeline = Pipeline(steps = [
        ("transformer", vectorizer),
        ("classifier", MLPClassifier(hidden_layer_sizes = 4, solver = "sgd", activation="relu",
                                     learning_rate_init=0.01, max_iter=50))
    ])

    pipeline.fit(X_train, y_train)

    tr_pred = pipeline.predict(X_train)
    tr_report = classification_report(y_train, tr_pred)
    print(f"Classification report::: {tr_report}")
    tr_confusion = confusion_matrix(y_train, tr_pred)


    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    report = classification_report(y_test, y_pred)
    print(f"Classification report::: {report}")
    confusion = confusion_matrix(y_test, y_pred)
    return {
        "Model": pipeline,
        "Train report": tr_report,
        "Train confusion matix": tr_confusion,
        "Test Report": report,
        "Test confusion": confusion}





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the customer data")
    parser.add_argument("--data", required=True, help="Path to raw customer CSV")
    parser.add_argument("--out", default="artifacts/model_mlp.pkl",
                         help="Where to save the trained model")
    args = parser.parse_args()

    result = train(args.data)
    save_artifact(result["Model"], args.out)
    print(f"Saved model to {args.out}")
    # Load and label
    # train("data/fake_or_real_news.csv")