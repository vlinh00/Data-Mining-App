import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def run_apriori(df, min_support=0.5, min_confidence=0.7):
    # df phải là one-hot encoded (0/1)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules
