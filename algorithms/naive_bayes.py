# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# def run_naive_bayes(df, target_column):
#     """
#     Chạy Naïve Bayes (Gaussian).
#     Input: df (DataFrame), target_column (string)
#     Output: accuracy, preds (DataFrame), fig (Confusion Matrix)
#     """

#     X = df.drop(columns=[target_column])
#     y = df[target_column]

#     # Encode categorical
#     for col in X.columns:
#         if X[col].dtype == "object":
#             X[col] = LabelEncoder().fit_transform(X[col])

#     if y.dtype == "object":
#         y = LabelEncoder().fit_transform(y)

#     # Chia train/test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     model = GaussianNB()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)

#     preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})

#     # Confusion Matrix chart
#     fig, ax = plt.subplots()
#     ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues", colorbar=False)
#     ax.set_title("Confusion Matrix - Naïve Bayes")

#     return acc, preds, fig

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def run_naive_bayes(df, target_column, use_laplace=False):
    """
    Chạy Naïve Bayes.
    - Nếu use_laplace=True: dùng CategoricalNB (Laplace smoothing).
    - Nếu use_laplace=False: dùng GaussianNB.
    Input: df (DataFrame), target_column (string)
    Output: accuracy, preds (DataFrame), fig (Confusion Matrix)
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if use_laplace:
        model = CategoricalNB(alpha=1.0)  # Laplace smoothing
    else:
        model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})

    # Confusion Matrix chart
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix - Naïve Bayes")

    return acc, preds, fig
