# import pandas as pd
# from sklearn.calibration import LabelEncoder
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# def run_naive_bayes(df, target_column):
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

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     model = GaussianNB()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     return acc, pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def run_naive_bayes(df, target_column):
    """
    Chạy Naïve Bayes (Gaussian).
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})

    # Confusion Matrix chart
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Naïve Bayes")

    return acc, preds, fig
