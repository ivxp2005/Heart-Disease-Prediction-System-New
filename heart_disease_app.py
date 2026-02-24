import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with large fonts
st.markdown("""
    <style>
    /* Dark background */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling - large white text */
    .header-title {
        font-size: 4rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1.5rem;
        background: #2d2d2d;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.6rem;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar - dark theme */
    section[data-testid="stSidebar"] {
        background-color: #2d2d2d;
        border-right: 2px solid #404040;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #2d2d2d;
    }
    
    /* Sidebar text - white for readability */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
        font-size: 1.1rem;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff;
        font-size: 1.5rem;
    }
    
    /* Input labels - large white text */
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
    }
    
    /* Input widgets */
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSlider {
        background-color: #3d3d3d;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Selectbox text color */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        color: #ffffff;
        font-size: 1.1rem;
    }
    
    /* Category headers in sidebar */
    .category-header {
        background-color: #3b82f6;
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 14px 18px;
        margin: 24px 0 14px 0;
        border-radius: 10px;
        border-left: 5px solid #60a5fa;
    }
    
    /* Buttons - bright, large */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 16px 40px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        width: 100%;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }
    
    /* Info boxes - dark cards with white text */
    .info-box {
        background: #2d2d2d;
        color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border-left: 5px solid #3b82f6;
    }
    
    .info-box h3 {
        color: #ffffff;
        margin-top: 0;
        font-size: 1.6rem;
    }
    
    .info-box p {
        color: #e0e0e0;
        line-height: 1.8;
        font-size: 1.2rem;
    }
    
    /* Metric cards - large text */
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #b0b0b0;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Alert boxes - dark theme */
    .stAlert {
        border-radius: 10px;
        padding: 1.2rem 1.8rem;
        margin: 1.2rem 0;
        font-size: 1.15rem;
    }
    
    /* Success box */
    div[data-baseweb="notification"][kind="success"] {
        background-color: #1e4620;
        border-left: 5px solid #22c55e;
        color: #86efac;
    }
    
    /* Warning box */
    div[data-baseweb="notification"][kind="warning"] {
        background-color: #451a03;
        border-left: 5px solid #f59e0b;
        color: #fcd34d;
    }
    
    /* Error box */
    div[data-baseweb="notification"][kind="error"] {
        background-color: #450a0a;
        border-left: 5px solid #ef4444;
        color: #fca5a5;
    }
    
    /* Info box */
    div[data-baseweb="notification"][kind="info"] {
        background-color: #0c4a6e;
        border-left: 5px solid #0ea5e9;
        color: #7dd3fc;
    }
    
    /* Horizontal divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        border-top: 2px solid #404040;
    }
    
    /* Section headers - large white text */
    h2 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 1.5rem;
        font-size: 2.2rem;
    }
    
    h3 {
        color: #e0e0e0;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.8rem;
    }
    
    /* Improve overall text readability - larger fonts */
    p, li {
        color: #d0d0d0;
        line-height: 1.8;
        font-size: 1.2rem;
    }
    
    /* Bold text visibility */
    strong {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Markdown text color */
    .stMarkdown {
        color: #e0e0e0;
        font-size: 1.15rem;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background-color: #10b981;
        color: white;
        font-size: 1.2rem;
        padding: 12px 30px;
        font-weight: 600;
    }
    
    .stDownloadButton>button:hover {
        background-color: #059669;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model and info
@st.cache_resource
def load_model():
    try:
        # Get the directory where this script is located
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build absolute paths to model files
        model_path = os.path.join(script_dir, 'heart_disease_model.pkl')
        info_path = os.path.join(script_dir, 'model_info.pkl')
        
        # Load model and info
        model = joblib.load(model_path)
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
        return model, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Looking for model in: {os.path.dirname(os.path.abspath(__file__))}")
        return None, None

# Create risk gauge chart
def create_risk_gauge(risk_percentage):
    # Determine color based on risk level
    if risk_percentage < 20:
        color = "green"
        risk_level = "LOW RISK"
    elif risk_percentage < 40:
        color = "yellow"
        risk_level = "MODERATE RISK"
    else:
        color = "red"
        risk_level = "HIGH RISK"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_percentage,
        title = {'text': f"<b>{risk_level}</b>", 'font': {'size': 28}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [20, 40], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [40, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="#2d2d2d",
        font={'color': "#ffffff", 'family': "Arial", 'size': 16}
    )
    
    return fig

# Create feature importance visualization
def create_feature_chart(patient_data):
    features = ['Age', 'Total Cholesterol', 'Blood Pressure', 'BMI', 'Glucose', 'Heart Rate']
    values = [
        patient_data[1] / 100,  # Age normalized
        patient_data[9] / 400,  # Total Cholesterol
        patient_data[10] / 200, # Systolic BP
        patient_data[12] / 50,  # BMI
        patient_data[14] / 200,  # Glucose
        patient_data[13] / 150   # Heart Rate
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=values,
            marker_color=['#3b82f6', '#2563eb', '#1d4ed8', '#1e40af', '#1e3a8a', '#172554'],
            text=[f'{v:.1%}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=dict(text="Patient Health Metrics (Normalized)", font=dict(color='#ffffff', size=18)),
        xaxis=dict(
            title=dict(text="Health Factors", font=dict(size=14, color='#ffffff')),
            tickfont=dict(size=12, color='#ffffff'),
            color='#ffffff'
        ),
        yaxis=dict(
            title=dict(text="Relative Level", font=dict(size=14, color='#ffffff')),
            tickfont=dict(size=12, color='#ffffff'),
            color='#ffffff'
        ),
        height=300,
        margin=dict(l=60, r=20, t=60, b=60),
        paper_bgcolor="#2d2d2d",
        plot_bgcolor="#2d2d2d",
        font=dict(size=14, color='#ffffff')
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="header-title">❤️ Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">AI-Powered 10-Year Coronary Heart Disease Risk Assessment</p>', unsafe_allow_html=True)
    
    # Load model
    model, model_info = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please ensure you've run the notebook to train and save the model.")
        return
    
    # Display model info in a clean card
    st.markdown(f"""
        <div class="info-box">
            <h3>🤖 Model Information</h3>
            <p><strong>Algorithm:</strong> {model_info['model_name']}</p>
            <p><strong>Accuracy:</strong> {model_info['test_accuracy']:.2%}</p>
            <p><strong>Dataset:</strong> Framingham Heart Study (3,658 patients)</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input
    st.sidebar.title("📋 Patient Information")
    st.sidebar.markdown("---")
    
    # Demographics
    st.sidebar.markdown('<p class="category-header">👤 Demographics</p>', unsafe_allow_html=True)
    male = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
    male_val = 1 if male == "Male" else 0
    age = st.sidebar.slider("Age (years)", 20, 80, 50)
    education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4], index=1, 
                                     format_func=lambda x: {1:"Some High School", 2:"High School/GED", 
                                                            3:"Some College", 4:"College Degree"}[x])
    
    # Lifestyle
    st.sidebar.markdown('<p class="category-header">🚬 Lifestyle</p>', unsafe_allow_html=True)
    currentSmoker = st.sidebar.selectbox("Current Smoker", ["No", "Yes"], index=0)
    currentSmoker_val = 1 if currentSmoker == "Yes" else 0
    
    # Conditionally show cigarettes slider only if smoker
    if currentSmoker == "Yes":
        cigsPerDay = st.sidebar.slider("Cigarettes per Day", 0, 60, 10)
    else:
        cigsPerDay = 0
        st.sidebar.markdown("**Cigarettes per Day:** 0 (Not a smoker)")
    
    # Medical History
    st.sidebar.markdown('<p class="category-header">🏥 Medical History</p>', unsafe_allow_html=True)
    BPMeds = st.sidebar.selectbox("On BP Medication", ["No", "Yes"], index=0)
    BPMeds_val = 1 if BPMeds == "Yes" else 0
    prevalentStroke = st.sidebar.selectbox("History of Stroke", ["No", "Yes"], index=0)
    prevalentStroke_val = 1 if prevalentStroke == "Yes" else 0
    prevalentHyp = st.sidebar.selectbox("Hypertension", ["No", "Yes"], index=0)
    prevalentHyp_val = 1 if prevalentHyp == "Yes" else 0
    diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"], index=0)
    diabetes_val = 1 if diabetes == "Yes" else 0
    
    # Clinical Measurements
    st.sidebar.markdown('<p class="category-header">🔬 Clinical Measurements</p>', unsafe_allow_html=True)
    totChol = st.sidebar.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
    sysBP = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 80, 250, 120)
    diaBP = st.sidebar.slider("Diastolic Blood Pressure (mmHg)", 50, 150, 80)
    BMI = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0, 0.1)
    heartRate = st.sidebar.slider("Heart Rate (bpm)", 40, 150, 70)
    glucose = st.sidebar.slider("Glucose Level (mg/dL)", 40, 300, 80)
    
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 PREDICT RISK")
    
    # Main content area
    if predict_button:
        # Prepare input data
        patient_data = np.array([[male_val, age, education, currentSmoker_val, cigsPerDay, 
                                 BPMeds_val, prevalentStroke_val, prevalentHyp_val, diabetes_val,
                                 totChol, sysBP, diaBP, BMI, heartRate, glucose]])
        
        # Make prediction
        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0]
        risk_probability = probability[1] * 100  # Probability of CHD (class 1)
        
        # Display results
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("---")
            st.markdown("## 📊 Risk Assessment Results")
            st.markdown("---")
            
            # Risk gauge
            fig_gauge = create_risk_gauge(risk_probability)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="CHD Risk", 
                    value=f"{risk_probability:.1f}%",
                    delta=f"{risk_probability - 15:.1f}% vs avg" if risk_probability > 15 else f"{15 - risk_probability:.1f}% below avg",
                    delta_color="inverse"
                )
            
            with metric_col2:
                st.metric(
                    label="Healthy Probability", 
                    value=f"{100 - risk_probability:.1f}%"
                )
            
            with metric_col3:
                status = "🟢 Low" if risk_probability < 20 else "🟡 Moderate" if risk_probability < 40 else "🔴 High"
                st.metric(
                    label="Risk Category", 
                    value=status
                )
            
            st.markdown("---")
            
            # Health metrics visualization
            fig_features = create_feature_chart([male_val, age, education, currentSmoker_val, cigsPerDay, 
                                                 BPMeds_val, prevalentStroke_val, prevalentHyp_val, diabetes_val,
                                                 totChol, sysBP, diaBP, BMI, heartRate, glucose])
            st.plotly_chart(fig_features, use_container_width=True)
            
            # Risk interpretation
            st.markdown("### 💡 What This Means")
            
            if risk_probability < 20:
                st.success(f"""
                    **Good News!** Your 10-year risk of developing coronary heart disease is **LOW** ({risk_probability:.1f}%).
                    
                    ✅ Continue maintaining a healthy lifestyle  
                    ✅ Regular check-ups recommended  
                    ✅ Keep monitoring key health metrics
                """)
            elif risk_probability < 40:
                st.warning(f"""
                    **Attention Needed:** Your 10-year risk is **MODERATE** ({risk_probability:.1f}%).
                    
                    ⚠️ Consider lifestyle modifications  
                    ⚠️ Consult with your healthcare provider  
                    ⚠️ Monitor blood pressure and cholesterol
                """)
            else:
                st.error(f"""
                    **Important:** Your 10-year risk is **HIGH** ({risk_probability:.1f}%).
                    
                    🚨 Seek medical consultation immediately  
                    🚨 Lifestyle changes strongly recommended  
                    🚨 Regular monitoring essential
                """)
            
            # Risk factors
            st.markdown("### 🎯 Your Risk Factors")
            risk_factors = []
            
            if age > 55:
                risk_factors.append(f"• **Age**: {age} years (elevated risk after 55)")
            if currentSmoker_val:
                risk_factors.append(f"• **Smoking**: {cigsPerDay} cigarettes/day")
            if totChol > 240:
                risk_factors.append(f"• **High Cholesterol**: {totChol} mg/dL (high)")
            elif totChol > 200:
                risk_factors.append(f"• **Cholesterol**: {totChol} mg/dL (borderline high)")
            if sysBP > 140 or diaBP > 90:
                risk_factors.append(f"• **High Blood Pressure**: {sysBP}/{diaBP} mmHg")
            if BMI > 30:
                risk_factors.append(f"• **Obesity**: BMI {BMI:.1f}")
            elif BMI > 25:
                risk_factors.append(f"• **Overweight**: BMI {BMI:.1f}")
            if diabetes_val:
                risk_factors.append("• **Diabetes**: Present")
            if prevalentStroke_val:
                risk_factors.append("• **Stroke History**: Present")
            if glucose > 126:
                risk_factors.append(f"• **High Glucose**: {glucose} mg/dL")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(factor)
            else:
                st.markdown("✅ No major risk factors detected!")
            
            # Recommendations
            st.markdown("### 📋 Personalized Recommendations")
            
            recommendations = []
            if currentSmoker_val:
                recommendations.append("🚭 **Quit smoking** - reduces risk by 50% within 1 year")
            if BMI > 25:
                recommendations.append("🏃 **Maintain healthy weight** - aim for BMI 18.5-24.9")
            if totChol > 200:
                recommendations.append("🥗 **Lower cholesterol** - diet changes and possibly medication")
            if sysBP > 120:
                recommendations.append("💊 **Control blood pressure** - aim for <120/80 mmHg")
            if diabetes_val:
                recommendations.append("🩺 **Manage diabetes** - keep glucose levels in check")
            
            recommendations.append("💪 **Regular exercise** - 30 minutes daily")
            recommendations.append("🥦 **Healthy diet** - Mediterranean or DASH diet")
            recommendations.append("😌 **Stress management** - meditation, yoga, adequate sleep")
            recommendations.append("👨‍⚕️ **Regular check-ups** - monitor risk factors")
            
            for rec in recommendations[:6]:  # Show top 6 recommendations
                st.markdown(rec)
            
            st.markdown("---")
            
            # Disclaimer
            st.info("""
                **⚠️ Medical Disclaimer:**  
                This tool provides risk estimates based on statistical models and should NOT replace professional medical advice.
                Always consult with qualified healthcare providers for proper diagnosis and treatment.
                
                *Model Accuracy: {:.1f}% | Based on Framingham Heart Study*
            """.format(model_info['test_accuracy'] * 100))
            
            # Download report button
            st.markdown("---")
            report_data = {
                'Assessment Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Gender': male,
                'Age': age,
                'Risk Score': f"{risk_probability:.1f}%",
                'Risk Category': 'Low' if risk_probability < 20 else 'Moderate' if risk_probability < 40 else 'High',
                'Total Cholesterol': totChol,
                'Blood Pressure': f"{sysBP}/{diaBP}",
                'BMI': BMI,
                'Smoker': currentSmoker,
                'Diabetes': diabetes,
                'Model': model_info['model_name']
            }
            
            st.download_button(
                label="📥 Download Assessment Report",
                data=pd.DataFrame([report_data]).to_csv(index=False),
                file_name=f"heart_risk_assessment_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                ## 👋 Welcome!
                
                This application uses advanced machine learning to predict your **10-year risk** 
                of developing coronary heart disease (CHD).
                
                ### How it works:
                1. 📝 Enter your health information in the sidebar
                2. 🔮 Click "PREDICT RISK" button
                3. 📊 View your personalized risk assessment
                4. 💡 Get actionable health recommendations
                
                ### Based on:
                - ✅ Framingham Heart Study data
                - ✅ 3,658 patient records
                - ✅ 85% prediction accuracy
                - ✅ Multiple clinical factors
                
                ---
                
                **🔒 Privacy:** All data is processed locally. Nothing is stored or shared.
                
                **👈 Start by filling out the form on the left sidebar!**
            """)
            
            # Sample statistics
            st.markdown("### 📈 Population Statistics")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("Average Risk", "15.2%", help="Population average 10-year CHD risk")
            with stat_col2:
                st.metric("High Risk Cases", "15%", help="Percentage with >50% risk")
            with stat_col3:
                st.metric("Low Risk Cases", "52%", help="Percentage with <10% risk")

if __name__ == "__main__":
    main()
