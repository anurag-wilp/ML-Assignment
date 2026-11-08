"""Synapse Atlas: model evaluation for the UCI Mice Protein dataset."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "model" / "catalog.joblib"
INK, MINT, GOLD, CORAL = "#16272a", "#27785e", "#e7c76b", "#ff8267"

st.set_page_config(page_title="Synapse Atlas | Protein models", page_icon="🧬",
                   layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
.stApp{background:#f0f2ec;color:#16272a;font-family:'DM Sans',sans-serif}
header,[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,.stDeployButton{display:none!important}
[data-testid="stHeader"]{background:#f0f2ec;height:0;min-height:0}
[data-testid="stTabs"] button{color:#16272a!important;background:#f0f2ec!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#16272a!important;border-bottom:2px solid #27785e!important}
[data-testid="stTabs"] button p{color:#16272a!important}
[data-testid="stSidebar"]{background:#16272a;color:#edf4e9}
[data-testid="stSidebar"] *{color:#edf4e9}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#16272a}
[data-testid="stMetric"]{background:#fff;border:1px solid #dfe5da;padding:14px 16px;border-radius:4px 15px 4px 4px}
[data-testid="stMetricLabel"]{font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.07em;font-size:.67rem;color:#536762}
[data-testid="stMetricValue"]{font-family:'Manrope',sans-serif;color:#16272a}
.mast{display:flex;justify-content:space-between;border-bottom:1px solid #cbd5ca;padding-bottom:12px;margin-bottom:22px}
.wordmark,.issue,.eyebrow,.section-label{font-family:'DM Mono',monospace}
.wordmark{letter-spacing:.12em;font-size:.74rem;color:#244a3d}
.issue{color:#536762;font-size:.68rem}
.hero{background:#16272a;color:#f3f4e9;border-radius:3px 24px 3px 3px;padding:25px 30px;position:relative;overflow:hidden;margin-bottom:22px}
.hero h1{font-family:'Manrope',sans-serif;font-size:2.2rem;letter-spacing:-.04em;margin:0 0 7px;color:#f3f4e9}
.hero p{color:#d1ddd5;font-size:.98rem;margin:0;max-width:590px}
.eyebrow,.section-label{font-size:.66rem;letter-spacing:.13em;text-transform:uppercase}
.eyebrow{color:#a5e3c2;margin-bottom:10px}
.orb{position:absolute;right:10%;top:17px;width:140px;height:140px;border:1px solid #a5e3c255;border-radius:50%;background:radial-gradient(circle at 36% 35%,#a5e3c244,transparent 62%);box-shadow:0 0 42px #a5e3c222,inset 0 0 28px #a5e3c211}
.orb:before,.orb:after{content:'';position:absolute;border:1px solid #ff826766;border-radius:50%;inset:18px 5px;transform:rotate(42deg)}
.orb:after{border-color:#e7c76b77;inset:7px 27px;transform:rotate(-38deg)}
.section-label{color:#536762;margin:16px 0 8px}
div[data-testid="stTabs"] button{font-family:'DM Mono',monospace;color:#16272a}
footer{visibility:hidden}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="mast"><div class="wordmark">◉ SYNAPSE ATLAS &nbsp;/&nbsp; PROTEIN CLASSIFICATION</div><div class="issue">UCI DATASET 342 · 8 GROUPS</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">Mouse brain · protein expression</div><h1>Protein profile classifier</h1><p>Evaluate saved models on labeled samples from the Mice Protein Expression study.</p><div class="orb"></div></div>', unsafe_allow_html=True)

if not CATALOG.exists():
    st.error("Saved models are missing. Run train_models.py to create the model artifacts.")
    st.stop()
catalog = joblib.load(CATALOG)
features, target, classes = catalog["features"], catalog["target"], catalog["classes"]
model_files = catalog["models"]

with st.sidebar:
    st.markdown("### EVALUATION")
    data_mode = st.radio("Test data", ["Sample holdout", "Upload CSV"])
    upload = st.file_uploader("Labeled test data", type="csv") if data_mode == "Upload CSV" else None
    selected = st.selectbox("Classifier", list(model_files))
    st.caption("CSV must include all protein feature columns and the class target.")

if data_mode == "Sample holdout":
    test_path = ROOT / "test_data.csv"
    if not test_path.exists():
        st.error("The sample holdout is missing.")
        st.stop()
    data = pd.read_csv(test_path)
    source_label = "Bundled sample holdout"
elif upload is None:
    st.info("Upload a labeled test CSV to view evaluation results.")
    st.stop()
else:
    try:
        data = pd.read_csv(upload)
        source_label = f"Uploaded: {upload.name}"
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()

missing = [column for column in features + [target] if column not in data.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()
if data.empty:
    st.warning("The uploaded CSV has no rows.")
    st.stop()
x, truth = data[features], data[target].astype(str)

@st.cache_resource
def load_estimator(filename):
    return joblib.load(ROOT / "model" / filename)

predictions, probability_map, rows = {}, {}, []
for name, filename in model_files.items():
    estimator = load_estimator(filename)
    pred, proba = estimator.predict(x), estimator.predict_proba(x)
    predictions[name], probability_map[name] = pred, proba
    try:
        auc = roc_auc_score(truth, proba, labels=estimator.classes_,
                            average="macro", multi_class="ovr")
    except ValueError:
        auc = np.nan
    rows.append({
        "Model": name, "Accuracy": accuracy_score(truth, pred), "AUC": auc,
        "Precision": precision_score(truth, pred, average="macro", zero_division=0),
        "Recall": recall_score(truth, pred, average="macro", zero_division=0),
        "F1": f1_score(truth, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(truth, pred),
    })
comparison = pd.DataFrame(rows).set_index("Model")
chosen = load_estimator(model_files[selected])
guess = predictions[selected]
chosen_scores = comparison.loc[selected]

st.markdown('<div class="section-label">EVALUATION METRICS</div>', unsafe_allow_html=True)
metric_columns = st.columns(6)
for column, label in zip(metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    value = chosen_scores[label]
    column.metric(label, "—" if pd.isna(value) else f"{value:.3f}")
st.caption(f"Source: {source_label} · {len(data):,} labeled samples · {selected} · macro averaged class metrics")

compare_tab, confusion_tab, report_tab = st.tabs(["Model comparison", "Confusion matrix", "Classification report"])
with compare_tab:
    st.markdown('<div class="section-label">ALL FIVE CLASSIFIERS · THIS TEST SET</div>', unsafe_allow_html=True)
    st.dataframe(comparison.style.format("{:.3f}")
                 .set_properties(**{"color":INK,"background-color":"#ffffff"})
                 .highlight_max(axis=0, color="#d9f1e1"),
                 use_container_width=True)
    chart_data = comparison.reset_index().melt(id_vars="Model",
        value_vars=["Accuracy", "AUC", "F1", "MCC"], var_name="Metric", value_name="Score")
    fig = px.bar(chart_data, x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_map={"Accuracy":MINT,"AUC":GOLD,"F1":CORAL,"MCC":INK},
        range_y=[0,1.04])
    fig.update_layout(height=360, margin=dict(l=5,r=5,t=15,b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans",size=11,color=INK),
        legend=dict(title="",font=dict(color=INK)), xaxis_title="", yaxis_title="Score",
        xaxis=dict(tickfont=dict(color=INK),linecolor="#9ba9a0"),
        yaxis=dict(tickfont=dict(color=INK),title_font=dict(color=INK),
                   gridcolor="#dfe5da",linecolor="#9ba9a0"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

with confusion_tab:
    matrix = confusion_matrix(truth, guess, labels=classes)
    heat = px.imshow(matrix, x=classes, y=classes, text_auto=True, aspect="auto",
        labels={"x":"Predicted class","y":"Actual class","color":"Samples"},
        color_continuous_scale=[[0,"#f0f2ec"],[.55,"#c8e5d4"],[1,"#91c9aa"]])
    heat.update_layout(height=500, margin=dict(l=5,r=5,t=15,b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans",size=11,color=INK),
        xaxis=dict(tickfont=dict(color=INK),title_font=dict(color=INK)),
        yaxis=dict(tickfont=dict(color=INK),title_font=dict(color=INK)),
        coloraxis_colorbar=dict(tickfont=dict(color=INK),title_font=dict(color=INK)))
    heat.update_traces(textfont=dict(color=INK))
    st.plotly_chart(heat, use_container_width=True, config={"displayModeBar":False})

with report_tab:
    report = pd.DataFrame(classification_report(truth, guess, labels=classes,
        output_dict=True, zero_division=0)).T
    report.index.name = "Class"
    st.dataframe(report.style.format("{:.3f}"), use_container_width=True)

st.caption("Models are loaded from saved artifacts. Uploaded rows are evaluated without retraining.")
