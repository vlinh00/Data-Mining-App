# import pandas as pd
# from mlxtend.frequent_patterns import apriori, association_rules

# def run_apriori(df, min_support=0.5, min_confidence=0.7):
#     # df phải là one-hot encoded (0/1)
#     frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
#     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
#     return rules

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

def run_apriori(df, min_support=0.5, min_confidence=0.7):
    """
    Chạy thuật toán Apriori.
    Input: df (one-hot encoded DataFrame, chỉ chứa 0/1)
    Output: rules (DataFrame), fig (biểu đồ support vs confidence)
    """

    # Ép kiểu về 0/1
    df = df.astype(bool).astype(int)

    # Tìm tập phổ biến
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # Sinh luật kết hợp
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Vẽ biểu đồ support - confidence
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(rules['support'], rules['confidence'], alpha=0.7, c=rules['lift'], cmap='viridis')
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Biểu đồ Support vs Confidence (màu = Lift)")
    fig.colorbar(ax.collections[0], ax=ax, label="Lift")

    return rules, fig
