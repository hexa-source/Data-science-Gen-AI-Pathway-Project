# Bank Customer Segmentor

Unsupervised + supervised customer segmentation for a Moniepoint-styled digital banking customer base — built to replace one-size-fits-all marketing and retention strategy with data-driven, behavior-based customer groups.

## Problem

Traditional marketing and retention strategies at many banks and fintechs treat customers as a single undifferentiated pool, or split them only by the crudest label available (e.g. "Business" vs "Individual"). This leads to:

- **Generic campaigns** that ignore real behavioral differences (transaction volume, login frequency, product usage, recency).
- **Missed cross-sell/upsell opportunities**, since product recommendations aren't matched to how a customer actually uses the platform.
- **No early warning system** for behavioral drift (falling transaction activity, rising failed transactions, declining logins) that typically precedes churn.

This project builds a segmentation system that groups customers by *behavior*, not just declared type, so marketing, product, and retention teams can act on data-backed segments instead of guesswork.

## Approach

The pipeline combines **unsupervised clustering** (to discover natural customer groupings) with a **supervised classification layer** (to assign new/incoming customers to a segment instantly, without re-running clustering every time).

1. **Data Quality Check & EDA**
   - Schema and null checks, duplicate checks, dropping redundant/erroneous columns.
   - Univariate analysis: customer type, regional/state/tier distribution, age, tenure, loan repayment rate, login frequency.
   - Correlation analysis across numeric features.

2. **Preprocessing & Feature Engineering**
   - Dropped missing value or redundant columns (`State`, `Occupation`, `Region`, duplicate average transaction-value column, etc.).
   - Balanced sampling: **25,000 Business + 25,000 Individual** customers drawn from the full 1,000,000-row synthetic dataset, for a 50,000-row modeling set.
   - `ColumnTransformer` pipeline: one-hot encoding for categorical fields (`Customer_type`, `Account_type`, `POS`, `Loan`, `Device`), ordinal encoding for `Tier`, and standard scaling for numeric behavioral features (transaction counts/values, balances, deposits/withdrawals, recency, login frequency, loan amount/repayment).

3. **Clustering (KMeans) — New, Regular & Premium Segments**
   - Silhouette score used to evaluate cluster quality across k = 2–10.
   - k = 3 selected as the primary segmentation, achieving a silhouette score of **~0.3048**.
   - The three resulting clusters are interpreted as **New**, **Regular** and **Premium** customer segments based on behavioral intensity (transaction volume/value, account balance, login frequency, product usage), not churn status.

4. **Supervised Classification Layer**
   - Cluster labels from KMeans are used as targets so the classifier *learns the pattern* behind the clustering, letting new customers be scored into a segment instantly without re-running KMeans each time.
   - classifiers used: MLPClassifier
   - 70/30 train-test split.

5. **Model Interpretability**
   - SHAP (`PermutationExplainer`) applied to the MLPClassifier model to explain which features drive segment assignment, including global summary/beeswarm plots and individual force plots.

## Dataset

- **Source**: Synthetic, Moniepoint-styled digital banking customer records (not real customer data).
- **Full size**: 1,000,000 rows.
- **Modeling sample**: 50,000 rows, balanced 25,000/25,000 across `Customer_type` (Business / Individual).
- **Key fields**: `Customer_type`, `Account_type`, `Tier`, `Age`, `Tenure`, `Monthly_txn_count`, `product_used`, `Total_txn_value`, `Avg_txn_value`, `Account_balance`, `Monthly_deposit`, `Monthly_withdrawal`, `POS` / `POS_transaction`, `Loan` / `Loan_amount` / `Loan_repay_rate`, `Device`, `Recency`, `Login_frequency`, `Failed_txt_count`.

## Results

| Model |Accuracy
|---|---|---|
| MLP Classifer | 100% |

The model achieve perfect accuracy at recovering the KMeans cluster labels.

> **Note on accuracy:** These figures reflect how well each classifier reconstructs the KMeans labels — not necessarily performance against an independent, ground-truth "true segment." Because `Customer_type` and closely related fields both informed the original clustering and remain available to the classifiers, the 100% accuracy is expected and should be read as *label recoverability*, not real-world classification difficulty. Anyone extending this project should watch for this pattern before treating these numbers as production-grade performance.



## Tech Stack

- **Modeling**: scikit-learn (KMeans, MLPClassifier), SHAP for interpretability.
- **Data analysis**: pandas, numpy, seaborn, matplotlib, plotly, statsmodels.



```
customer-segmentation/

                  # request/response schemas
├── data/
│   ├── processed/
│   └── raw/                          # moniepoint.csv (synthetic dataset)                  
├── notebooks/
│   └── customer_segmentation.ipynb           # preprocessing, clustering, classification, SHAP
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── cluster.py                    # KMeans clustering logic
│   ├── data_prep.py                  # data loading/cleaning
│   ├── predict.py                    # inference helpers
│   ├── preprocessor.py               # ColumnTransformer / encoding pipeline
│   ├── train.py                      # classifier training
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt