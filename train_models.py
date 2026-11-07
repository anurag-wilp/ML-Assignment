"""Train five classifiers on the UCI Mice Protein Expression dataset."""
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data_Cortex_Nuclear.xls"
DATA_URL = "https://archive.ics.uci.edu/static/public/342/mice+protein+expression.zip"
TARGET = "class"
SEED = 731


def read_data():
    """Load UCI's workbook, fetching it only when it is not present locally."""
    if not DATA_PATH.exists():
        archive_path = ROOT / "mice_protein_expression.zip"
        print("Downloading the UCI Mice Protein Expression dataset...")
        urlretrieve(DATA_URL, archive_path)
        from zipfile import ZipFile
        with ZipFile(archive_path) as archive:
            member = next(name for name in archive.namelist()
                          if name.endswith("Data_Cortex_Nuclear.xls"))
            DATA_PATH.write_bytes(archive.read(member))
        archive_path.unlink()
    raw = pd.read_excel(DATA_PATH)
    # Protein concentrations are numeric; isolate the eight experimental groups.
    feature_columns = [col for col in raw.columns if str(col).endswith("_N")]
    frame = raw[feature_columns].apply(pd.to_numeric, errors="coerce")
    frame[TARGET] = raw["class"].astype("string").str.strip()
    frame = frame.dropna(subset=[TARGET])
    return frame, raw.loc[frame.index, "MouseID"].astype(str)


def model_menu():
    prep = [SimpleImputer(strategy="median"), StandardScaler()]
    return {
        "Logistic Regression": make_pipeline(*prep, LogisticRegression(
            C=0.8, max_iter=3500, class_weight="balanced", random_state=SEED)),
        "Decision Tree": make_pipeline(SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=4,
                                   class_weight="balanced", random_state=SEED)),
        "K-Nearest Neighbors": make_pipeline(*prep,
            KNeighborsClassifier(n_neighbors=11, weights="distance")),
        "Gaussian Naive Bayes": make_pipeline(SimpleImputer(strategy="median"),
            GaussianNB(var_smoothing=1e-7)),
        "Support Vector Machine": make_pipeline(*prep,
            SVC(C=1.5, kernel="rbf", probability=True, class_weight="balanced",
                random_state=SEED)),
    }


def main():
    frame, groups = read_data()
    x, y = frame.drop(columns=TARGET), frame[TARGET].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    classes = sorted(y.unique())
    y_encoded = label_binarize(y_test, classes=classes)
    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    saved, results = {}, []
    for idx, (name, estimator) in enumerate(model_menu().items(), 1):
        estimator.fit(x_train, y_train)
        filename = f"protein_{idx}.joblib"
        joblib.dump(estimator, model_dir / filename)
        saved[name] = filename
        pred = estimator.predict(x_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_encoded, estimator.predict_proba(x_test),
                                 average="macro", multi_class="ovr"),
            "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1": f1_score(y_test, pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        })
    joblib.dump({"features": list(x.columns), "classes": classes,
                 "models": saved, "target": TARGET}, model_dir / "catalog.joblib")
    holdout = x_test.copy()
    holdout[TARGET] = y_test
    holdout.to_csv(ROOT / "test_data.csv", index=False)
    pd.DataFrame(results).to_csv(ROOT / "model_scores.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
