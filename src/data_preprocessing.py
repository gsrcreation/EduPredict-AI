import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "attendance", "study_hours", "assignment_score", "internal_marks",
    "previous_score", "backlogs", "participation"
]
TARGET_REGRESSION = "final_score"
TARGET_CLASSIFICATION = "risk_level"

def load_data(path="data/student_data.csv"):
    df = pd.read_csv(path)
    return df

def validate_student_input(data):
    ranges = {
        "attendance": (0, 100),
        "study_hours": (0, 24),
        "assignment_score": (0, 100),
        "internal_marks": (0, 100),
        "previous_score": (0, 100),
        "backlogs": (0, 20),
        "participation": (0, 10),
    }
    for key, (lo, hi) in ranges.items():
        if key not in data:
            raise ValueError(f"Missing field: {key}")
        value = float(data[key])
        if not (lo <= value <= hi):
            raise ValueError(f"{key} must be between {lo} and {hi}")
    return True

def prepare_data(df):
    clean = df.copy()
    clean[FEATURES] = clean[FEATURES].apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(subset=FEATURES + [TARGET_REGRESSION, TARGET_CLASSIFICATION])
    X = clean[FEATURES]
    y_reg = clean[TARGET_REGRESSION]
    y_cls = clean[TARGET_CLASSIFICATION]
    return X, y_reg, y_cls

def split_regression(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def split_classification(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)