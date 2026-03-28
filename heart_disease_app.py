import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
import time

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroGuard AI — Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load the trained model bundle ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load("heart_disease_model.pkl")
    return bundle

bundle   = load_model()
imputer  = bundle["imputer"]
scaler   = bundle["scaler"]
model    = bundle["model"]
threshold = bundle["threshold"]
feature_cols = bundle["feature_cols"]   # all 20 feature names

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }

section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
section[data-testid="stSidebar"] label { color: #8b949e !important; font-size:0.8rem !important;
    text-transform:uppercase; letter-spacing:0.08em; font-weight:600 !important; }

.stTabs [data-baseweb="tab-list"] {
    background:#161b22; border-radius:12px; padding:4px; gap:4px; border:1px solid #21262d; }
.stTabs [data-baseweb="tab"] {
    background:transparent; color:#8b949e; border-radius:8px;
    font-weight:500; font-size:0.9rem; padding:8px 20px; border:none; }
.stTabs [aria-selected="true"] { background:#1f6feb !important; color:#ffffff !important; }

.stSlider > div > div > div > div { background:#1f6feb; }
.stSelectbox > div > div { background-color:#161b22; border:1px solid #30363d;
    border-radius:8px; color:#e6edf3; }

.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#1f6feb,#388bfd); color:white;
    border:none; border-radius:10px; padding:14px 40px; font-size:1rem;
    font-weight:700; width:100%; letter-spacing:0.03em;
    box-shadow:0 4px 14px rgba(31,111,235,0.4); }
.stButton > button[kind="primary"]:hover { transform:translateY(-2px);
    box-shadow:0 6px 20px rgba(31,111,235,0.6); }

.ng-card { background:#161b22; border:1px solid #21262d; border-radius:14px;
    padding:22px 26px; margin-bottom:16px; }
.ng-metric { background:linear-gradient(135deg,#161b22,#0d1117); border:1px solid #21262d;
    border-radius:14px; padding:22px; text-align:center; }
.ng-metric .value { font-size:2.4rem; font-weight:700; line-height:1; margin-bottom:6px; }
.ng-metric .label { font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em; }

.ng-header { background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);
    border:1px solid #21262d; border-radius:16px; padding:28px 32px;
    margin-bottom:24px; text-align:center; }
.ng-header h1 { color:#388bfd; font-size:2rem; font-weight:700; margin:0 0 4px 0; }
.ng-header p  { color:#8b949e; font-size:0.95rem; margin:0; }

label { color:#c9d1d9 !important; font-weight:500 !important; font-size:0.9rem !important; }
hr { border-color:#21262d; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px;'>
        <span style='font-size:2.5rem;'>🫀</span>
        <h2 style='color:#388bfd;margin:8px 0 2px;font-size:1.1rem;'>NeuroGuard AI</h2>
        <p style='color:#8b949e;font-size:0.78rem;margin:0;'>Heart Disease Predictor</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    patient_id = st.text_input("🔖 Patient ID", value="P-2024-001")

    st.markdown("---")
    st.markdown("**🤖 Model Info**")
    import pickle
    with open("model_info.pkl", "rb") as f:
        minfo = pickle.load(f)
    st.caption(f"Type: Logistic Regression")
    st.caption(f"ROC-AUC: {minfo['roc_auc']:.2%}")
    st.caption(f"Threshold: {minfo['threshold']:.4f}")
    st.caption(f"Features: {len(feature_cols)}")

    st.markdown("---")
    st.caption(f"🟢 Online  •  v2.4.0")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("---")
    if st.button("🔄 Refresh"):
        st.rerun()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='ng-header'>
    <h1>🫀 NeuroGuard AI — Heart Disease Predictor</h1>
    <p>Framingham Heart Study · Logistic Regression · Threshold-Tuned</p>
</div>""", unsafe_allow_html=True)

# ── MODEL METRIC STRIP ────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f"""<div class='ng-metric'>
        <div class='value' style='color:#388bfd;'>{minfo['roc_auc']:.1%}</div>
        <div class='label'>ROC-AUC</div></div>""", unsafe_allow_html=True)
with s2:
    st.markdown(f"""<div class='ng-metric'>
        <div class='value' style='color:#3fb950;'>{minfo['test_accuracy']:.1%}</div>
        <div class='label'>Accuracy</div></div>""", unsafe_allow_html=True)
with s3:
    st.markdown(f"""<div class='ng-metric'>
        <div class='value' style='color:#d29922;'>{minfo['threshold']:.3f}</div>
        <div class='label'>Decision Threshold</div></div>""", unsafe_allow_html=True)
with s4:
    st.markdown(f"""<div class='ng-metric'>
        <div class='value' style='color:#8b949e;'>20</div>
        <div class='label'>Features Used</div></div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 🔍 Patient Assessment")
st.caption("Fill in all four tabs, then click **Predict CHD Risk**.")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "👤  Demographics",
    "🚬  Lifestyle",
    "🔬  Clinical Vitals",
    "🏥  Medical History",
])

# ─────────────────────────────────────────────────────
# TAB 1 · Demographics
#  Model features: male, age, education
# ─────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 👤 Basic Information")
        name      = st.text_input("Full Name", value="John Doe")
        age       = st.slider("Age", 18, 100, 52, help="Patient age in years")
        gender    = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox(
            "Education Level",
            ["Some High School", "High School / GED", "Some College", "College Degree"],
            index=1,
            help="Used by the Framingham model (1→4 scale)"
        )
    with c2:
        st.markdown("#### ℹ️ Why these matter")
        st.markdown("""
        <div class='ng-card' style='padding:16px 20px;'>
            <p style='color:#8b949e;font-size:0.9rem;margin:0;'>
            These three features are all direct Framingham Heart Study inputs:<br><br>
            • <b style='color:#e6edf3;'>Sex</b> — males have higher CHD risk<br>
            • <b style='color:#e6edf3;'>Age</b> — risk rises significantly after 50<br>
            • <b style='color:#e6edf3;'>Education</b> — proxy for lifestyle & health literacy
            </p>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# TAB 2 · Lifestyle
#  Model features: currentSmoker, cigsPerDay, BPMeds
# ─────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🚬 Smoking")
        smoking = st.selectbox(
            "Smoking Status",
            ["Never Smoked", "Former Smoker", "Current Smoker"]
        )
        if smoking == "Current Smoker":
            cigs_per_day = st.slider("Cigarettes per Day", 1, 60, 10)
        else:
            cigs_per_day = 0
            st.info("Cigarettes per day: **0** (not a current smoker)")

    with c2:
        st.markdown("#### 💊 Medication")
        bp_meds = st.selectbox(
            "On Blood Pressure Medication?",
            ["No", "Yes"],
            help="BPMeds is a direct model feature"
        )
        st.markdown("""<div class='ng-card' style='padding:14px 18px;margin-top:12px;'>
            <p style='color:#8b949e;font-size:0.88rem;margin:0;'>
            📌 <b style='color:#e6edf3;'>All three fields</b> feed directly into the
            trained model — currentSmoker, cigsPerDay, and BPMeds are Framingham features.
            </p></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# TAB 3 · Clinical Vitals
#  Model features: totChol, sysBP, diaBP, BMI, heartRate, glucose
# ─────────────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💓 Cardiovascular")
        sys_bp = st.slider("Systolic BP (mmHg)", 80, 250, 132,
                           help="sysBP — Framingham feature")
        dia_bp = st.slider("Diastolic BP (mmHg)", 50, 150, 83,
                           help="diaBP — Framingham feature | also used in pulse_pressure")
        heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 75,
                               help="heartRate — Framingham feature")
        tot_chol = st.slider("Total Cholesterol (mg/dL)", 100, 400, 234,
                             help="totChol — Framingham feature")
    with c2:
        st.markdown("#### ⚖️ Metabolic")
        bmi = st.slider("BMI", 15.0, 50.0, 26.5, 0.1,
                        help="BMI — Framingham feature")
        glucose = st.slider("Glucose Level (mg/dL)", 60, 300, 82,
                            help="glucose — Framingham feature")

        pulse_pressure = sys_bp - dia_bp
        current_smoker_val = 1 if smoking == "Current Smoker" else 0

        st.markdown("#### ⚡ Live Computed Features")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a2332,#161b22);border:1px solid #21262d;
                        border-radius:12px;padding:14px 18px;margin-bottom:10px;'>
                <div style='color:#58a6ff;font-size:0.7rem;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>
                    Pulse Pressure</div>
                <div style='color:#e6edf3;font-size:1.5rem;font-weight:700;line-height:1;'>
                    {pulse_pressure} <span style='font-size:0.85rem;color:#8b949e;font-weight:400;'>mmHg</span></div>
                <div style='color:#484f58;font-size:0.72rem;margin-top:4px;'>sysBP − diaBP</div>
            </div>
            <div style='background:linear-gradient(135deg,#1a2332,#161b22);border:1px solid #21262d;
                        border-radius:12px;padding:14px 18px;'>
                <div style='color:#58a6ff;font-size:0.7rem;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>
                    Smoker × Age</div>
                <div style='color:#e6edf3;font-size:1.5rem;font-weight:700;line-height:1;'>
                    {current_smoker_val * age}</div>
                <div style='color:#484f58;font-size:0.72rem;margin-top:4px;'>currentSmoker × age</div>
            </div>""", unsafe_allow_html=True)
        with e2:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a2332,#161b22);border:1px solid #21262d;
                        border-radius:12px;padding:14px 18px;margin-bottom:10px;'>
                <div style='color:#58a6ff;font-size:0.7rem;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>
                    Age × SysBP</div>
                <div style='color:#e6edf3;font-size:1.5rem;font-weight:700;line-height:1;'>
                    {age * sys_bp}</div>
                <div style='color:#484f58;font-size:0.72rem;margin-top:4px;'>age × sysBP</div>
            </div>
            <div style='background:linear-gradient(135deg,#1a2332,#161b22);border:1px solid #21262d;
                        border-radius:12px;padding:14px 18px;'>
                <div style='color:#58a6ff;font-size:0.7rem;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>
                    Smoking Burden</div>
                <div style='color:#e6edf3;font-size:1.5rem;font-weight:700;line-height:1;'>
                    {current_smoker_val * cigs_per_day}</div>
                <div style='color:#484f58;font-size:0.72rem;margin-top:4px;'>currentSmoker × cigsPerDay</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# TAB 4 · Medical History
#  Model features: prevalentStroke, prevalentHyp, diabetes
# ─────────────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏥 Conditions  *(direct Framingham features)*")
        hypertension = st.selectbox("Prevalent Hypertension", ["No", "Yes"],
                                    help="prevalentHyp")
        stroke_hist  = st.selectbox("History of Stroke",      ["No", "Yes"],
                                    help="prevalentStroke")
        diabetes     = st.selectbox("Diabetes",               ["No", "Yes"],
                                    help="diabetes")
    with c2:
        st.markdown("#### ℹ️ Note")
        st.markdown("""<div class='ng-card' style='padding:16px 20px;'>
            <p style='color:#8b949e;font-size:0.88rem;margin:0;'>
            Only these <b style='color:#e6edf3;'>3 conditions</b> are in the
            Framingham training data and therefore used by the model.<br><br>
            Other conditions (prior heart disease, hospitalizations, LDL flags)
            were <b style='color:#f85149;'>not in the training set</b> — adding
            them to the UI would be misleading.
            </p></div>""", unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
st.markdown("---")
bc1, bc2, bc3 = st.columns([1.5, 1, 1.5])
with bc2:
    predict = st.button("🫀  Predict CHD Risk", type="primary")

# ── RESULTS — USES THE ACTUAL MODEL ───────────────────────────────────────────
if predict:
    with st.spinner("Running model prediction…"):
        time.sleep(0.8)

    # ── Encode inputs to model values ────────────────────────────────────
    male_val      = 1 if gender == "Male" else 0
    edu_map       = {"Some High School": 1, "High School / GED": 2,
                     "Some College": 3, "College Degree": 4}
    edu_val       = edu_map[education]
    smoker_val    = 1 if smoking == "Current Smoker" else 0
    bp_meds_val   = 1 if bp_meds == "Yes" else 0
    stroke_val    = 1 if stroke_hist == "Yes" else 0
    hyp_val       = 1 if hypertension == "Yes" else 0
    diabetes_val  = 1 if diabetes == "Yes" else 0

    # ── Engineered features (same as train_model.py) ─────────────────────
    pulse_pressure_eng  = sys_bp - dia_bp
    age_sysBP_eng       = age * sys_bp
    smoke_age_eng       = smoker_val * age
    glucose_diabetes_eng= glucose * (diabetes_val + 1)
    smoking_burden_eng  = smoker_val * cigs_per_day

    # ── Build feature DataFrame (must match feature_cols order exactly) ──
    input_df = pd.DataFrame([[
        male_val, age, edu_val, smoker_val, cigs_per_day,
        bp_meds_val, stroke_val, hyp_val, diabetes_val,
        tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose,
        pulse_pressure_eng, age_sysBP_eng, smoke_age_eng,
        glucose_diabetes_eng, smoking_burden_eng
    ]], columns=feature_cols)

    # ── Run through the exact same pipeline used during training ─────────
    X_proc = scaler.transform(imputer.transform(input_df))
    chd_prob  = model.predict_proba(X_proc)[0][1]  # probability of CHD
    prediction = int(chd_prob >= threshold)          # 0 = No CHD, 1 = CHD
    risk_pct   = chd_prob * 100

    risk_label  = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
    risk_color  = "#f85149" if prediction == 1 else "#3fb950"

    st.success("✅  Prediction complete!")
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    # ── Top result cards ──────────────────────────────────────────────────
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""<div class='ng-metric'>
            <div class='value' style='color:{risk_color};'>{risk_pct:.1f}%</div>
            <div class='label'>CHD Probability</div></div>""", unsafe_allow_html=True)
    with r2:
        outcome = "CHD Predicted" if prediction == 1 else "No CHD"
        st.markdown(f"""<div class='ng-metric'>
            <div class='value' style='color:{risk_color};font-size:1.4rem;'>
            {"⚠️" if prediction == 1 else "✅"}</div>
            <div class='label'>{outcome}</div></div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""<div class='ng-metric'>
            <div class='value' style='color:#d29922;'>{threshold:.3f}</div>
            <div class='label'>Decision Threshold</div></div>""", unsafe_allow_html=True)
    with r4:
        conf_gap = abs(chd_prob - threshold) * 100
        st.markdown(f"""<div class='ng-metric'>
            <div class='value' style='color:#8b949e;'>{conf_gap:.1f}%</div>
            <div class='label'>Margin from Threshold</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Probability gauge ────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={"text": "10-Year CHD Probability (%)",
                   "font": {"color": "#c9d1d9", "size": 14}},
            number={"font": {"color": risk_color, "size": 52},
                    "suffix": "%"},
            gauge={
                "axis":  {"range": [0, 100], "tickcolor": "#8b949e",
                          "tickfont": {"color": "#8b949e"}},
                "bar":   {"color": risk_color, "thickness": 0.7},
                "bgcolor": "#161b22",
                "bordercolor": "#21262d",
                "steps": [
                    {"range": [0,  threshold*100], "color": "rgba(63,185,80,0.12)"},
                    {"range": [threshold*100, 100], "color": "rgba(248,81,73,0.12)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100
                }
            }
        ))
        gauge.update_layout(
            height=320, paper_bgcolor="#161b22",
            font={"color": "#c9d1d9"},
            margin=dict(l=30, r=30, t=60, b=20)
        )
        st.plotly_chart(gauge, width="stretch")
        st.caption(f"⚪ White line = decision threshold ({threshold*100:.1f}%)")

    with ch2:
        # Feature contribution bar (model coefficients × scaled input)
        st.subheader("📊 Feature Contributions")
        coefs = model.coef_[0]
        # Use the scaled input for contribution
        contrib = coefs * X_proc[0]
        # Show top 10 by absolute value
        feature_labels = [
            "Sex (Male)", "Age", "Education", "Current Smoker", "Cigs/Day",
            "BP Meds", "Stroke History", "Hypertension", "Diabetes",
            "Total Cholesterol", "Systolic BP", "Diastolic BP", "BMI",
            "Heart Rate", "Glucose",
            "Pulse Pressure", "Age×SysBP", "Smoker×Age",
            "Glucose×Diabetes", "Smoker×Cigs"
        ]
        top_idx  = np.argsort(np.abs(contrib))[-10:][::-1]
        top_vals = contrib[top_idx]
        top_lbls = [feature_labels[i] for i in top_idx]

        bar_colors = ["#f85149" if v > 0 else "#3fb950" for v in top_vals]
        bar_fig = go.Figure(go.Bar(
            x=top_vals, y=top_lbls,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in top_vals],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=10)
        ))
        bar_fig.update_layout(
            title=dict(text="Top 10 Feature Contributions (Red = ↑ Risk)",
                       font=dict(color="#c9d1d9", size=13)),
            xaxis=dict(color="#8b949e", showgrid=True, gridcolor="#21262d",
                       zeroline=True, zerolinecolor="white"),
            yaxis=dict(color="#c9d1d9"),
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            height=320,
            margin=dict(l=10, r=60, t=40, b=20),
            font=dict(color="#c9d1d9")
        )
        st.plotly_chart(bar_fig, width="stretch")
        st.caption("🔴 Red = increases CHD risk  |  🟢 Green = decreases CHD risk")

    st.markdown("---")

    # ── Clinical recommendation ───────────────────────────────────────────
    st.subheader("💡 Clinical Recommendations")
    recs = []
    if prediction == 1:
        recs.append(("🏥", "High 10-year CHD risk detected",
                     "Cardiology referral strongly recommended"))
    else:
        recs.append(("✅", "Low 10-year CHD risk",
                     "Routine annual cardiovascular check-up recommended"))

    if smoker_val == 1:
        recs.append(("🚭", f"Active smoker ({cigs_per_day} cigs/day)",
                     "Smoking cessation reduces CHD risk by ~50% within 1 year"))
    if sys_bp > 140 or dia_bp > 90:
        recs.append(("💊", f"Hypertension ({sys_bp}/{dia_bp} mmHg)",
                     "BP control target: < 130/80 mmHg; discuss medication with physician"))
    if tot_chol > 240:
        recs.append(("🩺", f"High cholesterol ({tot_chol} mg/dL)",
                     "Statin therapy & dietary modification recommended"))
    if bmi > 30:
        recs.append(("⚖️", f"Obesity (BMI {bmi:.1f})",
                     "Structured weight loss: 5-10% reduction lowers CHD risk"))
    if glucose > 126:
        recs.append(("🩸", f"Elevated glucose ({glucose} mg/dL)",
                     "Diabetes management — HbA1c target < 7%"))

    recs += [
        ("🏃", "Physical activity",
         "≥150 min/week moderate aerobic exercise strengthens the heart"),
        ("🥗", "Diet", "Mediterranean/DASH diet — clinically proven cardiovascular benefit"),
    ]

    for emoji_s, title, detail in recs[:6]:
        st.markdown(f"""
        <div class='ng-card' style='padding:14px 20px;margin-bottom:10px;'>
            <span style='font-size:1.3rem;'>{emoji_s}</span>
            <strong style='color:#e6edf3;margin-left:8px;'>{title}</strong>
            <span style='color:#8b949e;margin-left:6px;'>— {detail}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Summary table + download ──────────────────────────────────────────
    st.subheader("📄 Patient Report")
    summary = {
        "Patient ID":       patient_id,
        "Name":             name,
        "Age":              age,
        "Gender":           gender,
        "Education":        education,
        "Smoking Status":   smoking,
        "Cigs/Day":         cigs_per_day,
        "BP Medication":    bp_meds,
        "Systolic BP":      f"{sys_bp} mmHg",
        "Diastolic BP":     f"{dia_bp} mmHg",
        "Pulse Pressure":   f"{pulse_pressure_eng} mmHg",
        "Heart Rate":       f"{heart_rate} bpm",
        "Total Cholesterol":f"{tot_chol} mg/dL",
        "BMI":              f"{bmi:.1f}",
        "Glucose":          f"{glucose} mg/dL",
        "Hypertension":     hypertension,
        "Stroke History":   stroke_hist,
        "Diabetes":         diabetes,
        "CHD Probability":  f"{risk_pct:.2f}%",
        "Prediction":       "CHD Risk" if prediction == 1 else "No CHD",
        "Model Threshold":  f"{threshold:.4f}",
        "Assessment Date":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    st.dataframe(
        pd.DataFrame([(k, str(v)) for k, v in summary.items()], columns=["Field", "Value"]),
        width="stretch", hide_index=True
    )

    dl1, dl2, dl3 = st.columns([1.5, 1, 1.5])
    with dl2:
        st.download_button(
            label="📥  Download Report (CSV)",
            data=pd.DataFrame([summary]).to_csv(index=False),
            file_name=f"chd_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

else:
    st.markdown("""
    <div class='ng-card' style='text-align:center;padding:40px;'>
        <h2 style='color:#388bfd;margin-top:0;'>🫀 Welcome to NeuroGuard AI</h2>
        <p style='color:#8b949e;font-size:1rem;max-width:600px;margin:0 auto 20px;'>
            Fill in the patient's details across the <strong style='color:#e6edf3;'>4 tabs</strong>
            above, then click <strong style='color:#e6edf3;'>🫀 Predict CHD Risk</strong>.<br><br>
            Predictions are made by the <b style='color:#388bfd;'>trained Logistic Regression model</b>
            using all 15 Framingham features + 5 engineered interaction terms.
        </p>
    </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#8b949e;font-size:0.78rem;padding:10px 0 20px;'>
    🫀 <strong style='color:#388bfd;'>NeuroGuard AI</strong> &nbsp;·&nbsp;
    Framingham Heart Study Model &nbsp;·&nbsp;
    ⚠️ For clinical decision support only — not a substitute for professional medical judgment.
</div>""", unsafe_allow_html=True)
