

import pandas as pd

from Delivarable1.src.utils import load_artifact

FEATURE_ORDER = [
    'Customer_type', 'Account_type',
    'POS', 'Loan', 'Device', 'Tier', 'Age', 'Tenure',
    'Monthly_txn_count', 'product_used', 'Total_txn_value', 'Avg_txn_valu',
    'Avg_txn_value', 'Account_balance', 'Monthly_deposit',
    'Monthly_withdrawal', 'POS_transaction', 'Recency', 'Login_frequency',
    'Loan_amount', 'Loan_repay_rate',
    'Failed_txt_count'
    ]

def predict_one(model_path: str, record: dict) -> dict:
   
    model = load_artifact(model_path)
    X = pd.DataFrame([record], columns=FEATURE_ORDER)
    prediction = model.predict(X)[0]
    return {
        "prediction": int(prediction)}


if __name__ == "__main__":
    example = {
        'Customer_type': "Individual", 'Account_type': "personal account",'POS': "No", 'Loan': "No",
        'Device': "mobile app", 'Tier': "tier 2", 'Age': 35, 'Tenure': 60,
        'Monthly_txn_count': 90, 'product_used': 6, 'Total_txn_value': 100000,
        'Avg_txn_value': 100000, 'Account_balance': 300000, 'Monthly_deposit': 200000,
        'Monthly_withdrawal': 197000, 'POS_transaction': 100, 'Recency': 100, 'Login_frequency': 90,
        'Loan_amount': 100000, 'Loan_repay_rate': 0.87,
        'Failed_txt_count': 10
        }
    result = predict_one("artifacts/mlp_model", example)
    print(result)

