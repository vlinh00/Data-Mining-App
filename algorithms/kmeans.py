# import pandas as pd
# from sklearn.cluster import KMeans

# def run_kmeans(df, n_clusters=3):
#     model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
#     df['cluster'] = model.fit_predict(df)
#     return df, model

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_kmeans(df, n_clusters=3):
    """
    Chạy K-means clustering.
    Input: df (DataFrame numeric), n_clusters
    Output: clustered df, model, fig (scatter plot nếu 2D)
    """

    # Chỉ nhận numeric
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError("K-means cần ít nhất 2 cột số để vẽ scatter plot.")

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = model.fit_predict(numeric_df)

    # Scatter plot (chỉ vẽ với 2D)
    fig = None
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=numeric_df.iloc[:, 0],
            y=numeric_df.iloc[:, 1],
            hue=df['cluster'],
            palette="Set2",
            ax=ax,
            s=80
        )
        ax.set_xlabel(numeric_df.columns[0])
        ax.set_ylabel(numeric_df.columns[1])
        ax.set_title("K-means Clustering")

    return df, model, fig
