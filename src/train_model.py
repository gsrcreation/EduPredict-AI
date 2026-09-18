from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from .data_preprocessing import load_data, prepare_data, FEATURES

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def train_all(data_path="data/student_data.csv"):
    df = load_data(data_path)
    X, y_reg, y_cls = prepare_data(df)

    Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=.2, random_state=42)
    regressors = {
        "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=250, random_state=42)
    }
    reg_results = []
    for name, model in regressors.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        reg_results.append({
            "model": name,
            "MAE": mean_absolute_error(yte, pred),
            "RMSE": mean_squared_error(yte, pred) ** 0.5,
            "R2": r2_score(yte, pred)
        })
    best_reg = min(reg_results, key=lambda x: x["RMSE"])
    reg_model = regressors[best_reg["model"]]
    import joblib
    joblib.dump(reg_model, MODEL_DIR/"performance_model.pkl")

    Xtr, Xte, ytr, yte = train_test_split(X, y_cls, test_size=.2, random_state=42, stratify=y_cls)
    classifiers = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=250, random_state=42)
    }
    cls_results = []
    for name, model in classifiers.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        cls_results.append({
            "model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, average="weighted", zero_division=0),
            "Recall": recall_score(yte, pred, average="weighted", zero_division=0),
            "F1": f1_score(yte, pred, average="weighted", zero_division=0)
        })
    best_cls = max(cls_results, key=lambda x: x["F1"])
    cls_model = classifiers[best_cls["model"]]
    joblib.dump(cls_model, MODEL_DIR/"risk_model.pkl")

    pd.DataFrame(reg_results).to_csv(MODEL_DIR/"regression_results.csv", index=False)
    pd.DataFrame(cls_results).to_csv(MODEL_DIR/"classification_results.csv", index=False)
    return best_reg, best_cls

if __name__ == "__main__":
    print(train_all())