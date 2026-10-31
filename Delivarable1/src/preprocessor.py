import pandas as pd
import argparse

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from Delivarable1.src.data_prep import tiers_cat, cat_features, num_features, drop_redundancies, load_data


data = "data/processed/bank_customer(1).csv"

def preprocessor(X: str):
    df = load_data(X)
    preprocessed = ColumnTransformer(transformers = [
        ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_features),
        ("tier", OrdinalEncoder(), tiers_cat),
        ("num", StandardScaler(), num_features)
        ])
    df_transformed = preprocessed.fit_transform(df)
    feature_names = preprocessed.get_feature_names_out()
    new_df = pd.DataFrame(df_transformed, columns = feature_names, index = df.index)
    new_df.reset_index(inplace = True)
    new_df = pd.concat([new_df.drop(columns = ["index", "cat__Customer_type_Business", "cat__Account_type_personal account", "cat__POS_Yes", "cat__Loan_Yes",
                       "cat__Device_pos"]), df["Label"]], axis = 1)
    print(new_df)
    return new_df
if __name__ == "__main__":
    preprocessor(data)

def feature_names(X: str):
    df = load_data(X)
    preprocessed = ColumnTransformer(transformers = [
        ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_features),
        ("tier", OrdinalEncoder(), tiers_cat),
        ("num", StandardScaler(), num_features)
        ])
    df_transformed = preprocessed.fit_transform(df)
    feature_names = preprocessed.get_feature_names_out()
    return feature_names

def pipeline(X: str):
    df = load_data(X)
    transformer = ColumnTransformer(transformers = [
        ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_features),
        ("tier", OrdinalEncoder(), tiers_cat),
        ("num", StandardScaler(), num_features)
        ])
    pipeline = Pipeline(steps = [
        ("transformer", transformer),
        ("classifier", LogisticRegression())
        ])
    df_transformed = pipeline.fit_transform(df)
    feat_names = feature_names()
    new_df = pd.DataFrame(df_transformed, columns = feat_names, index = df.index)
    new_df.reset_index(inplace = True)
    new_df = new_df.drop(columns = "index")
    return new_df
