# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# import graphviz

# def run_decision_tree(df, target_column):
#     # Tách dữ liệu
#     X = df.drop(columns=[target_column])
#     y = df[target_column]

#     # Encode tất cả các cột chuỗi
#     le_dict = {}
#     for col in X.columns:
#         if X[col].dtype == "object":
#             le = LabelEncoder()
#             X[col] = le.fit_transform(X[col])
#             le_dict[col] = le  # lưu lại nếu muốn decode sau

#     if y.dtype == "object":
#         le_y = LabelEncoder()
#         y = le_y.fit_transform(y)

#     # Huấn luyện cây quyết định
#     model = DecisionTreeClassifier(criterion="entropy")
#     model.fit(X, y)

#     y_pred = model.predict(X)
#     acc = accuracy_score(y, y_pred)

#     # Xuất cây
#     dot_data = export_graphviz(model, out_file=None,
#                                feature_names=X.columns,
#                                class_names=[str(c) for c in set(y)],
#                                filled=True, rounded=True,
#                                special_characters=True)

#     graph = graphviz.Source(dot_data)
#     return acc, graph

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import graphviz

def run_decision_tree_id3(
    df,
    target_column,
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=42,
):
    """
    Thuật toán ID3 (criterion='entropy') với tham số tùy chỉnh
    """
    # Tách dữ liệu
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode dữ liệu chuỗi
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Train model
    model = DecisionTreeClassifier(
        criterion="entropy",   # ID3 = entropy
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state
    )
    model.fit(X, y)

    # Accuracy
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    # Xuất cây
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=X.columns,
        class_names=[str(c) for c in set(y)],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    return acc, graph

