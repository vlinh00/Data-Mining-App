import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans(df, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df['cluster'] = model.fit_predict(df)
    return df, model
