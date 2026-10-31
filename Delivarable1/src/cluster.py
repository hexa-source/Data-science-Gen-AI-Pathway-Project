
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Delivarable1.src.preprocessor import preprocessor

data = "data/processed/bank_customer.csv"
def segment_engine(data):
    print("\n==========Clustering Algorithm for Labels Identification==========\n")
    print("loading data...")
    time.sleep(2)
    print("Prepraring data for preprocessing........................................................")
    time.sleep(2)
    print("loading OneHotEncoder for cat features..........................................................")
    print("loading OrdinalEncoder for tier feature...........................................................")
    print("loading StandardScalar for num features..............................................................")
    time.sleep(5)
    print("data preprocessing successful...................................................")
    print("converting preprocessed data to pd_DataFrame >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    time.sleep(3)
    

    df = preprocessor(data)
    time.sleep(2)
    print("Dropping redundant preprocessed column.........................................................")
    print("Replacing preprocessed get_feature_names with original data feature names...........................")
    
    time.sleep(3)
    print("Preprocessing Successfull.............................................................................")
    
    time.sleep(2)
    print("Preparing preprocessed data for clustering process....................................................")
    print("loading KMeans Clustering Model::::....n_cluster=3...........................................................")
    time.sleep(1)
    km = KMeans(n_clusters = 3)
    t1 = time.time()
    print("Fitting data into the model........................................................................")
    km = km.fit(df)
    print("Clustering successfully")
    clustering_time = time.time()-t1
    print(f"Clustering Model Training Time: {clustering_time:.3f} seconds")
    print("Silhoutte Score Evaluation...")
    sil = silhouette_score(df, km.labels_)
    print(f"Silhouttte Score: {sil}")
    print("Appending cluster.labels_ to the raw data...")
    df["labels"] = km.labels_
    print("Data appended successfully")
    print("\n-----Preprocessed data with label column in pandas dataframe-----\n")
    print(f"data: {df}")
    return {"df": df}
if __name__ == "__main__":
    segment_engine(data)
