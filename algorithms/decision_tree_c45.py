# # # import pandas as pd
# # # from sklearn.tree import DecisionTreeClassifier, export_graphviz
# # # from sklearn.metrics import accuracy_score
# # # from sklearn.preprocessing import LabelEncoder
# # # import graphviz

# # # def run_decision_tree_c45(df, target_column):
# # #     # Tách dữ liệu
# # #     X = df.drop(columns=[target_column])
# # #     y = df[target_column]

# # #     # Encode tất cả các cột chuỗi
# # #     le_dict = {}
# # #     for col in X.columns:
# # #         if X[col].dtype == "object":
# # #             le = LabelEncoder()
# # #             X[col] = le.fit_transform(X[col])
# # #             le_dict[col] = le

# # #     if y.dtype == "object":
# # #         le_y = LabelEncoder()
# # #         y = le_y.fit_transform(y)

# # #     # Cây quyết định C4.5 (mô phỏng bằng entropy trong sklearn)
# # #     model = DecisionTreeClassifier(criterion="entropy", splitter="best")
# # #     model.fit(X, y)

# # #     y_pred = model.predict(X)
# # #     acc = accuracy_score(y, y_pred)

# # #     # Xuất cây
# # #     dot_data = export_graphviz(model, out_file=None,
# # #                                feature_names=X.columns,
# # #                                class_names=[str(c) for c in set(y)],
# # #                                filled=True, rounded=True,
# # #                                special_characters=True)

# # #     graph = graphviz.Source(dot_data)
# # #     return acc, graph

# # import pandas as pd
# # import math
# # from graphviz import Digraph

# # # ====== Hàm tính toán cơ bản ======
# # def entropy(col):
# #     counts = col.value_counts(normalize=True)
# #     return -sum(p * math.log2(p) for p in counts if p > 0)

# # def info_gain_ratio(df, attr, target):
# #     total_entropy = entropy(df[target])
# #     values = df[attr].unique()

# #     weighted_entropy = 0
# #     split_info = 0
# #     for v in values:
# #         subset = df[df[attr] == v]
# #         p = len(subset) / len(df)
# #         weighted_entropy += p * entropy(subset[target])
# #         if p > 0:
# #             split_info -= p * math.log2(p)

# #     info_gain = total_entropy - weighted_entropy
# #     if split_info == 0:
# #         return 0
# #     return info_gain / split_info

# # # ====== Xây cây C4.5 ======
# # def build_tree(df, target):
# #     # Nếu tất cả nhãn giống nhau
# #     if len(df[target].unique()) == 1:
# #         return df[target].iloc[0]

# #     # Nếu không còn thuộc tính để chia
# #     if len(df.columns) == 1:
# #         return df[target].mode()[0]

# #     # Chọn thuộc tính tốt nhất theo Gain Ratio
# #     attrs = [c for c in df.columns if c != target]
# #     gains = {attr: info_gain_ratio(df, attr, target) for attr in attrs}
# #     best_attr = max(gains, key=gains.get)

# #     tree = {best_attr: {}}
# #     for v in df[best_attr].unique():
# #         subset = df[df[best_attr] == v].drop(columns=[best_attr])
# #         if subset.empty:
# #             tree[best_attr][v] = df[target].mode()[0]
# #         else:
# #             tree[best_attr][v] = build_tree(subset, target)
# #     return tree

# # # ====== Dự đoán ======
# # def predict(tree, sample):
# #     if not isinstance(tree, dict):
# #         return tree
# #     attr = next(iter(tree))
# #     branches = tree[attr]
# #     value = sample.get(attr)
# #     if value in branches:
# #         return predict(branches[value], sample)
# #     else:
# #         # Nếu giá trị chưa thấy trong training -> chọn nhãn phổ biến nhất trong nhánh
# #         return max(branches.values(), key=lambda x: isinstance(x, dict))

# # def accuracy(tree, df, target):
# #     preds = df.apply(lambda row: predict(tree, row), axis=1)
# #     return (preds == df[target]).mean()

# # # ====== Vẽ Graphviz ======
# # def dict_to_graphviz(tree, dot=None, parent=None, edge_label=""):
# #     if dot is None:
# #         dot = Digraph()

# #     if isinstance(tree, dict):
# #         for attr, branches in tree.items():
# #             node_id = str(id(tree))
# #             dot.node(node_id, attr, shape="ellipse", style="filled", color="#E6F0FA")
# #             if parent:
# #                 dot.edge(parent, node_id, label=edge_label)
# #             for v, sub in branches.items():
# #                 dict_to_graphviz(sub, dot, node_id, str(v))
# #     else:
# #         leaf_id = str(id(tree)) + "_leaf"
# #         dot.node(leaf_id, str(tree), shape="box", style="filled", color="lightgreen")
# #         if parent:
# #             dot.edge(parent, leaf_id, label=edge_label)
# #     return dot

# # # ====== Hàm chính gọi từ app.py ======
# # def run_decision_tree_c45(df, target_column):
# #     tree = build_tree(df, target_column)
# #     acc = accuracy(tree, df, target_column)
# #     graph = dict_to_graphviz(tree)
# #     return acc, graph

# import pandas as pd
# import numpy as np
# import math
# from graphviz import Digraph

# # ====== Hàm tính toán ======
# def entropy(col):
#     counts = col.value_counts(normalize=True)
#     return -sum(p * math.log2(p) for p in counts if p > 0)

# def info_gain_ratio(df, attr, target):
#     total_entropy = entropy(df[target])
#     values = df[attr].unique()

#     weighted_entropy = 0
#     split_info = 0
#     for v in values:
#         subset = df[df[attr] == v]
#         p = len(subset) / len(df)
#         weighted_entropy += p * entropy(subset[target])
#         if p > 0:
#             split_info -= p * math.log2(p)

#     info_gain = total_entropy - weighted_entropy
#     if split_info == 0:
#         return 0
#     return info_gain / split_info

# # ====== Xây cây C4.5 ======
# def build_tree(df, target, depth=0, max_depth=None, min_samples_split=2):
#     # Thông tin node
#     node = {
#         "samples": len(df),
#         "value": df[target].value_counts().to_dict(),
#         "entropy": entropy(df[target]),
#     }

#     # Dừng nếu chỉ còn 1 lớp
#     if len(df[target].unique()) == 1:
#         node["label"] = df[target].iloc[0]
#         return node

#     # Dừng nếu hết thuộc tính hoặc điều kiện dừng
#     if len(df.columns) == 1 or (max_depth is not None and depth >= max_depth) or len(df) < min_samples_split:
#         node["label"] = df[target].mode()[0]
#         return node

#     # Chọn thuộc tính tốt nhất theo Gain Ratio
#     attrs = [c for c in df.columns if c != target]
#     gains = {attr: info_gain_ratio(df, attr, target) for attr in attrs}
#     best_attr = max(gains, key=gains.get)

#     node["attribute"] = best_attr
#     node["children"] = {}

#     for v in df[best_attr].unique():
#         subset = df[df[best_attr] == v].drop(columns=[best_attr])
#         if subset.empty:
#             node["children"][v] = {"label": df[target].mode()[0], "samples": 0, "value": {}, "entropy": 0}
#         else:
#             node["children"][v] = build_tree(
#                 subset, target, depth + 1, max_depth, min_samples_split
#             )

#     return node

# # ====== Dự đoán ======
# def predict(tree, sample):
#     if "attribute" not in tree:
#         return tree.get("label")
#     attr = tree["attribute"]
#     value = sample.get(attr)
#     if value in tree["children"]:
#         return predict(tree["children"][value], sample)
#     return max(tree["value"], key=tree["value"].get)

# def accuracy(tree, df, target):
#     preds = df.apply(lambda row: predict(tree, row), axis=1)
#     return (preds == df[target]).mean()

# # ====== Vẽ Graphviz ======
# def tree_to_graphviz(tree, dot=None, parent=None, edge_label="", show_entropy=True, show_value=True):
#     if dot is None:
#         dot = Digraph()
#         dot.attr("node", shape="box", style="rounded,filled", color="lightblue2", fontname="Arial")

#     # Xây label node
#     if "attribute" in tree:
#         label = f"{tree['attribute']}"
#     else:
#         label = f"Class = {tree['label']}"

#     if show_entropy:
#         label += f"\nentropy = {tree['entropy']:.2f}"
#     label += f"\nsamples = {tree['samples']}"
#     if show_value:
#         label += f"\nvalue = {list(tree['value'].values())}"

#     node_id = str(id(tree))
#     dot.node(node_id, label)

#     if parent:
#         dot.edge(parent, node_id, label=edge_label)

#     if "children" in tree:
#         for v, sub in tree["children"].items():
#             tree_to_graphviz(sub, dot, node_id, str(v), show_entropy, show_value)

#     return dot

# # ====== Hàm chính ======
# def run_decision_tree_c45(
#     df, target_column, max_depth=None, min_samples_split=2, show_entropy=True, show_value=True
# ):
#     tree = build_tree(df, target_column, max_depth=max_depth, min_samples_split=min_samples_split)
#     acc = accuracy(tree, df, target_column)
#     graph = tree_to_graphviz(tree, show_entropy=show_entropy, show_value=show_value)
#     return acc, graph

import math
from graphviz import Digraph

def entropy(col):
    counts = col.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in counts if p > 0)

def info_gain_ratio(df, attr, target):
    total_entropy = entropy(df[target])
    values = df[attr].unique()
    weighted_entropy, split_info = 0, 0
    for v in values:
        subset = df[df[attr] == v]
        p = len(subset) / len(df)
        weighted_entropy += p * entropy(subset[target])
        if p > 0:
            split_info -= p * math.log2(p)
    info_gain = total_entropy - weighted_entropy
    return 0 if split_info == 0 else info_gain / split_info

def build_tree(df, target, depth=0, max_depth=None, min_samples_split=2):
    node = {
        "samples": len(df),
        "value": df[target].value_counts().to_dict(),
        "entropy": entropy(df[target]),
    }
    if len(df[target].unique()) == 1:
        node["label"] = df[target].iloc[0]; return node
    if len(df.columns) == 1 or (max_depth and depth >= max_depth) or len(df) < min_samples_split:
        node["label"] = df[target].mode()[0]; return node

    attrs = [c for c in df.columns if c != target]
    gains = {a: info_gain_ratio(df, a, target) for a in attrs}
    best_attr = max(gains, key=gains.get)
    node["attribute"] = best_attr
    node["children"] = {}
    for v in df[best_attr].unique():
        subset = df[df[best_attr] == v].drop(columns=[best_attr])
        if subset.empty:
            node["children"][v] = {"label": df[target].mode()[0], "samples": 0, "value": {}, "entropy": 0}
        else:
            node["children"][v] = build_tree(subset, target, depth+1, max_depth, min_samples_split)
    return node

def predict(tree, sample):
    if "attribute" not in tree:
        return tree.get("label")
    attr = tree["attribute"]
    value = sample.get(attr)
    if value in tree["children"]:
        return predict(tree["children"][value], sample)
    return max(tree["value"], key=tree["value"].get)

def accuracy(tree, df, target):
    preds = df.apply(lambda row: predict(tree, row), axis=1)
    return (preds == df[target]).mean()

def tree_to_graphviz(tree, dot=None, parent=None, edge_label="", show_entropy=True, show_value=True):
    if dot is None:
        dot = Digraph()
        dot.attr("node", shape="box", style="rounded,filled", color="lightblue2", fontname="Arial")
    label = f"{tree.get('attribute', 'Class = ' + str(tree.get('label')))}"
    if show_entropy: label += f"\nentropy = {tree['entropy']:.2f}"
    label += f"\nsamples = {tree['samples']}"
    if show_value: label += f"\nvalue = {list(tree['value'].values())}"
    node_id = str(id(tree))
    dot.node(node_id, label)
    if parent: dot.edge(parent, node_id, label=edge_label)
    if "children" in tree:
        for v, sub in tree["children"].items():
            tree_to_graphviz(sub, dot, node_id, str(v), show_entropy, show_value)
    return dot

def run_decision_tree_c45(df, target_column, max_depth=None, min_samples_split=2, show_entropy=True, show_value=True):
    tree = build_tree(df, target_column, max_depth=max_depth, min_samples_split=min_samples_split)
    acc = accuracy(tree, df, target_column)
    graph = tree_to_graphviz(tree, show_entropy=show_entropy, show_value=show_value)
    return acc, graph
