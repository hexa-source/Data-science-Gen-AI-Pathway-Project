

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def data_summary(df: pd.DataFrame) -> dict:
    """Return a quick data-quality snapshot: shape, info, nulls, duplicates."""
    return {
        "shape": df.shape,
        "info": df.info(),
        "nulls": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "customer_type_counts": df["Customer_type"].value_counts().to_dict(),
    }

def drop_redundancies(df: pd.DataFrame):
    df = df.drop(columns = df.drop(
        columns = ["State", "Occupation", "Region", "loan_repay_rate", "Business", "Avg_txn_valu"],
        inplace = True))
    return df


def balance_customer_record(df: pd.DataFrame, customer_type: str = "Customer_type",
                     business: int = 25_000, individual: int = 25_000,
                     random_state: int = 42) -> pd.DataFrame:
    """
    sample of 250_000 from each customer type to maintain customer type balance.
    """
    business = df[df["Customer_type"]== "Business"].sample(n = 25000, random_state = 42)
    individual = df[df["Customer_type"]== "Individual"].sample(n = 25000, random_state = 42)
    balanced = pd.concat([business, individual]).reset_index(drop=True)
    return balanced


def split_features_target(df: pd.DataFrame, labels: str = "Label"):
    """Split a dataframe into features (X) and target (Y)."""
    X = df.drop(columns=labels)
    Y = df[labels]
    return X, Y

tiers_cat = ["Tier"]

cat_features = [
    'Customer_type', 
    'Account_type', 'POS',
    'Loan', 'Device'
    ] 
num_features = [
    'Age', 'Tenure', 'Monthly_txn_count',
    'product_used', 'Total_txn_value', 'Avg_txn_value', 'Account_balance',
    'Monthly_deposit', 'Monthly_withdrawal', 'POS_transaction', 'Recency',
    'Login_frequency', 'Loan_amount', 'Loan_repay_rate',
    'Failed_txt_count'
    ]
