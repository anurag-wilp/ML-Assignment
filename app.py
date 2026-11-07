"""BeanScope: inspect saved models on a labeled Dry Bean holdout CSV."""
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "model" / "catalog.joblib"

st.set_page_config(page_title="BeanScope | Variety lab", page_icon="🫘", layout="wide")
st.markdown("""<style>
.stApp {background: radial-gradient(circle at 82% 0%, #e9f2dd 0, #f7f5ee 42%, #f5f1e7 100%);}
[data-testid="stMetric"] {background:#fffdf7;border:1px solid #dedbcf;padding:14px;border-radius:14px;}
.hero {padding:1.2rem 1.5rem;background:#203d30;color:#f8f4e8;border-radius:18px;margin-bottom:1rem}
.hero p {color:#cbd9c9;margin-bottom:0}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🧬 Synapse Atlas</h1><p>Can protein signatures reveal a mouse experimental group?</p></div>',
            unsafe_allow_html=True)

if not CATALOG_PATH.exists():
    st.error("Model files are missing. Run `python train_models.py` once to prepare the saved classifiers and test_data.csv.")
    st.stop()

catalog = joblib.load(CATALOG_PATH)
model_names = list(catalog["models"])
left, right = st.columns([1, 1.4])
with left:
    chosen = st.selectbox("Choose a classifier", model_names)
    uploaded = st.file_uploader("Upload labeled test rows (CSV)", type="csv")
    st.caption("The CSV must contain the protein measurements and a `class` label.")
    if st.button("Use the supplied holdout", use_container_width=True):
        st.session_state["use_sample"] = True
with right:
    st.markdown("#### A small field guide")
    st.write("Each row contains protein expression measurements from mouse brain tissue. Choose a saved classifier, then compare its predictions with the known experimental group.")

source = uploaded if uploaded is not None else (ROOT / "test_data.csv" if st.session_state.get("use_sample") else None)
if source is None:
    st.info("Upload your labeled CSV or choose the supplied holdout to see the model’s scorecard.")
    st.stop()
try:
    frame = pd.read_csv(source)
except Exception as exc:
    st.error(f"Could not read that CSV: {exc}")
    st.stop()
target = catalog["target"]
features = catalog["features"]
missing = [column for column in features + [target] if column not in frame.columns]
if missing:
    st.error("Required columns missing: " + ", ".join(missing))
    st.stop()
x, truth = frame[features], frame[target].astype(str)
estimator = joblib.load(ROOT / "model" / catalog["models"][chosen])
guess = estimator.predict(x)
probabilities = estimator.predict_proba(x)
try:
    auc = roc_auc_score(truth, probabilities, labels=estimator.classes_,
                        average="macro", multi_class="ovr")
except ValueError:
    auc = float("nan")
scores = [
    ("Accuracy", accuracy_score(truth, guess)),
    ("Macro AUC", auc),
    ("Precision", precision_score(truth, guess, average="macro", zero_division=0)),
    ("Recall", recall_score(truth, guess, average="macro", zero_division=0)),
    ("F1", f1_score(truth, guess, average="macro", zero_division=0)),
    ("MCC", matthews_corrcoef(truth, guess)),
]
st.subheader(f"{chosen} · {len(frame):,} labeled samples")
metrics = st.columns(6)
for column, (label, value) in zip(metrics, scores):
    column.metric(label, "—" if pd.isna(value) else f"{value:.3f}")
matrix = confusion_matrix(truth, guess, labels=catalog["classes"])
cm, report = st.tabs(["Confusion map", "Class by class"])
with cm:
    fig = px.imshow(matrix, x=catalog["classes"], y=catalog["classes"],
                    labels={"x": "Predicted variety", "y": "Known variety", "color": "Beans"},
                    color_continuous_scale=["#f4ecdb", "#76956c", "#203d30"], text_auto=True,
                    aspect="auto")
    fig.update_layout(margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(fig, use_container_width=True)
with report:
    st.dataframe(pd.DataFrame(classification_report(truth, guess, output_dict=True,
        zero_division=0)).T.style.format("{:.3f}"), use_container_width=True)
st.caption("Models are loaded from saved artifacts. Uploaded rows are evaluated without retraining.")
