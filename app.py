"""Synapse Atlas: a protein assay bench for exploring mouse expression profiles."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "model" / "catalog.joblib"
INK, MINT, CORAL, GOLD = "#16272a", "#a5e3c2", "#ff8267", "#e7c76b"

st.set_page_config(page_title="Synapse Atlas | Assay bench", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;600;700&family=Manrope:wght@600;700;800&display=swap');
.stApp{background:#f0f2ec;color:#16272a;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#16272a;color:#edf4e9}
[data-testid="stSidebar"] *{color:#edf4e9}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#16272a}
[data-testid="stMetric"]{background:#fff;border:1px solid #dfe5da;padding:14px 16px;border-radius:4px 15px 4px 4px;box-shadow:0 4px 16px #182d2510}
[data-testid="stMetricLabel"]{font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:.08em;font-size:.67rem;color:#607470}
[data-testid="stMetricValue"]{font-family:'Manrope',sans-serif;color:#16272a}
.mast{display:flex;justify-content:space-between;border-bottom:1px solid #cbd5ca;padding-bottom:13px;margin-bottom:24px}
.wordmark,.issue,.eyebrow,.section-label,.tiny{font-family:'DM Mono',monospace}
.wordmark{letter-spacing:.14em;font-size:.75rem;font-weight:500;color:#244a3d}
.issue,.tiny{color:#73827a;font-size:.68rem}
.hero{background:#16272a;color:#f3f4e9;border-radius:3px 28px 3px 3px;padding:28px 32px;position:relative;overflow:hidden;min-height:210px;margin-bottom:23px}
.hero h1{font-family:'Manrope',sans-serif;font-size:2.45rem;letter-spacing:-.045em;margin:0 0 8px;color:#f3f4e9}
.hero p{color:#b9ccc0;font-size:1rem;margin:0;max-width:560px}
.eyebrow,.section-label{font-size:.66rem;letter-spacing:.14em;text-transform:uppercase}
.eyebrow{color:#a5e3c2;margin-bottom:12px}
.orb{position:absolute;right:10%;top:20px;width:166px;height:166px;border:1px solid #a5e3c255;border-radius:50%;background:radial-gradient(circle at 36% 35%,#a5e3c244,transparent 62%);box-shadow:0 0 50px #a5e3c222,inset 0 0 30px #a5e3c211}
.orb:before,.orb:after{content:'';position:absolute;border:1px solid #ff826766;border-radius:50%;inset:20px 5px;transform:rotate(42deg)}
.orb:after{border-color:#e7c76b77;inset:7px 31px;transform:rotate(-38deg)}
.section-label{color:#607470;margin:10px 0}
.panel{background:#fff;border:1px solid #dfe5da;border-radius:4px 16px 4px 4px;padding:17px}
.panel h3{font-family:'Manrope',sans-serif;margin:4px 0;color:#16272a;font-size:1.05rem}
.stButton>button{background:#a5e3c2;color:#16272a;border:0;border-radius:4px 12px;font-weight:700}
footer{visibility:hidden}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="mast"><div class="wordmark">◉ SYNAPSE ATLAS &nbsp;/&nbsp; ASSAY BENCH</div><div class="issue">UCI · DATASET 342 &nbsp;/&nbsp; FIELD NOTE 01</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">Mouse brain · protein expression · 8 experimental groups</div><h1>Read the molecular weather.</h1><p>Explore how protein signatures separate genotype, treatment, and behavior groups in the Mice Protein Expression study.</p><div class="orb"></div></div>', unsafe_allow_html=True)

if not CATALOG.exists():
    st.error("Saved models are missing. Run train_models.py to create the assay artifacts.")
    st.stop()
catalog = joblib.load(CATALOG)
features, target, class_names = catalog["features"], catalog["target"], catalog["classes"]
model_files = catalog["models"]

with st.sidebar:
    st.markdown("### ASSAY CONTROLS")
    st.markdown('<div class="tiny">CHOOSE A RUN</div>', unsafe_allow_html=True)
    source_mode = st.radio("Data source", ["Reference holdout", "Upload a CSV"], label_visibility="collapsed")
    upload = st.file_uploader("Labeled sample sheet", type="csv") if source_mode == "Upload a CSV" else None
    st.markdown('<div class="tiny">MODEL LENS</div>', unsafe_allow_html=True)
    selected = st.selectbox("Classifier", list(model_files), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Study key**")
    st.caption("Each row is a mouse tissue measurement. Labels combine genotype, treatment, and behavior. The reference split keeps each mouse entirely on one side to avoid repeated-measure leakage.")
    st.markdown('<div class="tiny">77 PROTEIN FEATURES · 8 LABELS · 5 CLASSIFIERS</div>', unsafe_allow_html=True)

if source_mode == "Reference holdout":
    path = ROOT / "test_data.csv"
    if not path.exists():
        st.error("The reference holdout is missing.")
        st.stop()
    data = pd.read_csv(path)
elif upload is None:
    st.info("Choose a labeled CSV in the assay controls to start a run.")
    st.stop()
else:
    try:
        data = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read this sample sheet: {exc}")
        st.stop()
missing = [col for col in features + [target] if col not in data.columns]
if missing:
    st.error("Sample sheet is missing required columns: " + ", ".join(missing))
    st.stop()
if data.empty:
    st.warning("This sample sheet has no rows.")
    st.stop()
x, truth = data[features], data[target].astype(str)

@st.cache_resource
def load_estimator(filename):
    return joblib.load(ROOT / "model" / filename)

predictions, probability_map, result_rows = {}, {}, []
for name, filename in model_files.items():
    estimator = load_estimator(filename)
    prediction, probability = estimator.predict(x), estimator.predict_proba(x)
    predictions[name], probability_map[name] = prediction, probability
    try:
        auc = roc_auc_score(truth, probability, labels=estimator.classes_,
                            average="macro", multi_class="ovr")
    except ValueError:
        auc = np.nan
    result_rows.append({
        "Classifier": name,
        "Accuracy": accuracy_score(truth, prediction),
        "Macro AUC": auc,
        "Precision": precision_score(truth, prediction, average="macro", zero_division=0),
        "Recall": recall_score(truth, prediction, average="macro", zero_division=0),
        "Macro F1": f1_score(truth, prediction, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(truth, prediction),
    })
scores = pd.DataFrame(result_rows).set_index("Classifier")
estimator = load_estimator(model_files[selected])
guess, probabilities, chosen = predictions[selected], probability_map[selected], scores.loc[selected]

st.markdown('<div class="section-label">01 — RUN SNAPSHOT</div>', unsafe_allow_html=True)
metrics = st.columns([1.05, 1, 1, 1, 1])
metrics[0].metric("Samples in run", f"{len(data):,}")
metrics[1].metric("Current lens", selected.split()[0])
metrics[2].metric("Accuracy", f"{chosen['Accuracy']:.3f}")
metrics[3].metric("Macro F1", f"{chosen['Macro F1']:.3f}")
metrics[4].metric("MCC", f"{chosen['MCC']:.3f}")

field, detail = st.tabs(["MODEL FIELD NOTES", "CONFUSION & CLASS DETAIL"])
with field:
    chart_col, notes_col = st.columns([1.35, .85], gap="large")
    with chart_col:
        st.markdown('<div class="section-label">02 — CLASSIFIER COMPARISON</div>', unsafe_allow_html=True)
        melted = scores.reset_index().melt(id_vars="Classifier", value_vars=["Macro F1", "Macro AUC", "MCC"], var_name="Signal", value_name="Score")
        comparison = px.bar(melted, x="Score", y="Classifier", color="Signal", barmode="group", orientation="h",
            color_discrete_map={"Macro F1":"#27785e","Macro AUC":GOLD,"MCC":CORAL},
            category_orders={"Classifier":list(model_files)[::-1]}, range_x=[0,1.04])
        comparison.update_layout(height=350,margin=dict(l=0,r=8,t=8,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono",size=10,color=INK),legend_title_text="",xaxis_title="Score",yaxis_title="",
            xaxis=dict(showgrid=True,gridcolor="#e7ebe3",zeroline=False))
        st.plotly_chart(comparison,use_container_width=True,config={"displayModeBar":False})
        st.dataframe(scores.style.format("{:.3f}").highlight_max(axis=0,color="#d9f1e1"),use_container_width=True)
    with notes_col:
        leader = scores["Macro F1"].idxmax()
        st.markdown('<div class="section-label">THE RUN LEADER</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><div class="tiny">BEST MACRO F1</div><h3>{leader}</h3><p style="color:#607470">{scores.loc[leader,"Macro F1"]:.3f} F1 · {scores.loc[leader,"MCC"]:.3f} MCC</p></div>',unsafe_allow_html=True)
        st.markdown('<div class="section-label" style="margin-top:20px">SELECTED MODEL · SIX METRICS</div>',unsafe_allow_html=True)
        metric_pairs=[("Accuracy","Accuracy"),("AUC · OVR","Macro AUC"),("Precision","Precision"),("Recall","Recall"),("F1","Macro F1"),("MCC","MCC")]
        for i in range(0,6,2):
            a,b=st.columns(2)
            a.metric(metric_pairs[i][0],f"{chosen[metric_pairs[i][1]]:.3f}")
            b.metric(metric_pairs[i+1][0],f"{chosen[metric_pairs[i+1][1]]:.3f}")
        st.markdown('<div class="tiny">AUC is macro one-vs-rest; other class metrics use macro averaging.</div>',unsafe_allow_html=True)

with detail:
    heat_col, report_col = st.columns([1.2,.8],gap="large")
    with heat_col:
        st.markdown('<div class="section-label">03 — ERROR TOPOLOGY</div>',unsafe_allow_html=True)
        matrix=confusion_matrix(truth,guess,labels=class_names)
        heat=px.imshow(matrix,x=class_names,y=class_names,text_auto=True,aspect="auto",
            labels={"x":"Predicted group","y":"Observed group","color":"Samples"},
            color_continuous_scale=[[0,"#f0f2ec"],[.45,MINT],[1,INK]])
        heat.update_layout(height=440,margin=dict(l=0,r=0,t=8,b=0),paper_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Mono",size=10,color=INK))
        st.plotly_chart(heat,use_container_width=True,config={"displayModeBar":False})
    with report_col:
        st.markdown('<div class="section-label">04 — GROUP RESOLUTION</div>',unsafe_allow_html=True)
        report=pd.DataFrame(classification_report(truth,guess,labels=class_names,output_dict=True,zero_division=0)).T
        st.dataframe(report.loc[class_names,["precision","recall","f1-score","support"]].style.format({"precision":"{:.2f}","recall":"{:.2f}","f1-score":"{:.2f}","support":"{:.0f}"}),use_container_width=True,height=420)

st.markdown('<div class="section-label">05 — SINGLE SAMPLE INSPECTOR</div>',unsafe_allow_html=True)
row_number=st.slider("Select a sample from this run",0,len(data)-1,0,format="row %d")
sample_left,sample_mid,sample_right=st.columns([.8,1,1.3],gap="large")
observed=truth.iloc[row_number]
predicted=guess[row_number]
sample_probs=probabilities[row_number]
ranking=np.argsort(sample_probs)[::-1][:5]
with sample_left:
    st.markdown(f'<div class="panel"><div class="tiny">SAMPLE {row_number+1:03d} / {len(data):03d}</div><h3 style="margin-top:10px">{"Signal matched" if predicted==observed else "Signal diverged"}</h3><p style="color:#607470">Observed<br><b style="color:{INK}">{observed}</b></p><p style="color:#607470">Predicted<br><b style="color:{INK}">{predicted}</b></p></div>',unsafe_allow_html=True)
with sample_mid:
    prob_frame=pd.DataFrame({"Group":[estimator.classes_[i] for i in ranking],"Confidence":[sample_probs[i] for i in ranking]})
    confidence=px.bar(prob_frame,x="Confidence",y="Group",orientation="h",range_x=[0,1],text=prob_frame["Confidence"].map(lambda v:f"{v:.1%}"),color_discrete_sequence=["#27785e"])
    confidence.update_layout(title="Top group probabilities",height=245,margin=dict(l=0,r=5,t=35,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Mono",size=10,color=INK),xaxis_title="",yaxis_title="",showlegend=False)
    confidence.update_traces(textposition="outside")
    st.plotly_chart(confidence,use_container_width=True,config={"displayModeBar":False})
with sample_right:
    signals=x.iloc[row_number]
    strongest=signals.abs().nlargest(8).sort_values()
    protein=go.Figure(go.Bar(x=strongest.values,y=strongest.index,orientation="h",marker_color=[CORAL if value<0 else "#27785e" for value in strongest.values]))
    protein.update_layout(title="Largest-magnitude protein signals",height=245,margin=dict(l=0,r=5,t=35,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Mono",size=9,color=INK),xaxis_title="Expression value",yaxis_title="")
    st.plotly_chart(protein,use_container_width=True,config={"displayModeBar":False})

st.markdown('<div class="tiny" style="text-align:right;padding:20px 0 5px">SYNAPSE ATLAS / SAVED CLASSIFIERS · NO RETRAINING ON UPLOAD</div>',unsafe_allow_html=True)
