import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_naive_bayes(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode tất cả các cột chuỗi
    le_dict = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le  # lưu lại nếu muốn decode sau

    if y.dtype == "object":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return acc, pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
