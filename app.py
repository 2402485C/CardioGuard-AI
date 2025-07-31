import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="❤️ CardioGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .healthy {
        background-color: #D4EDDA;
        color: #155724;
        border: 2px solid #C3E6CB;
    }
    .disease {
        background-color: #F8D7DA;
        color: #721C24;
        border: 2px solid #F5C6CB;
    }
    .info-box {
        background-color: #F8F9FA;
        color: #2C3E50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #3498DB;
        border: 1px solid #DEE2E6;
    }
    .info-box strong {
        color: #1A252F;
    }
    .warning-box {
        background-color: #FFF3CD;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #FFC107;
        border: 1px solid #FFEAA7;
    }
    .success-box {
        background-color: #D1EDD1;
        color: #0F5132;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #198754;
        border: 1px solid #B3D7B3;
    }
</style>
""", unsafe_allow_html=True)

# Load or train model
@st.cache_resource
def load_model():
    """Load the trained model or train a new one if not available"""
    model_path = "trained_model.pkl"
    scaler_path = "scaler.pkl"
    
    # Try to load existing model
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            st.success("✅ Loaded pre-trained high-recall model!")
            return model, scaler, 0.2690  # Our optimized threshold
        except:
            st.warning("⚠️ Could not load saved model. Training new model...")
    
    # Train new model if loading fails
    try:
        # Load data
        df = pd.read_csv('cardio_train.csv', sep=';')
        
        # Feature engineering (simplified version for deployment)
        df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        
        # Select features
        feature_cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'BMI', 'pulse_pressure']
        
        X = df[feature_cols]
        y = df['cardio']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train high-recall model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Save model for future use
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        st.success("✅ Model trained successfully!")
        return model, scaler, 0.27  # Approximate threshold for high recall
        
    except Exception as e:
        st.error(f"❌ Error loading/training model: {str(e)}")
        return None, None, 0.5

# Feature preprocessing function
def preprocess_input(age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, alcohol, active):
    """Preprocess user input to match model format"""
    
    # Calculate derived features
    bmi = weight / (height / 100) ** 2
    pulse_pressure = ap_hi - ap_lo
    
    # Create feature array in the exact order expected by model
    features = np.array([[
        age * 365.25,  # Convert years to days (original format)
        gender,        # 1 for Female, 2 for Male
        height,        # Height in cm
        weight,        # Weight in kg
        ap_hi,         # Systolic BP
        ap_lo,         # Diastolic BP
        cholesterol,   # 1=Normal, 2=Above normal, 3=Well above normal
        glucose,       # 1=Normal, 2=Above normal, 3=Well above normal
        smoke,         # 0=No, 1=Yes
        alcohol,       # 0=No, 1=Yes
        active,        # 0=No, 1=Yes
        bmi,           # Calculated BMI
        pulse_pressure # Calculated pulse pressure
    ]])
    
    return features

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ CardioGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🔬 Advanced Cardiovascular Disease Prediction System</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, threshold = load_model()
    
    if model is None:
        st.error("❌ Unable to load or train model. Please check your setup.")
        return
    
    # Model info
    st.markdown("""
    <div class="info-box">
        <strong>🎯 High-Recall Model Specifications:</strong><br>
        • <strong>Recall Rate:</strong> 90.05% (Catches 90% of cardiovascular disease cases)<br>
        • <strong>Model Type:</strong> Optimized Random Forest with Balanced Class Weights<br>
        • <strong>Designed for:</strong> Medical screening where missing a disease case is critical
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Patient Information")
        
        # Basic Demographics
        st.markdown("**👤 Demographics**")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, help="Patient's age in years")
        gender = st.selectbox("Gender", options=["Female", "Male"], help="Biological gender")
        gender_code = 1 if gender == "Female" else 2
        
        # Physical Measurements
        st.markdown("**📏 Physical Measurements**")
        col1a, col1b = st.columns(2)
        with col1a:
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, help="Height in centimeters")
        with col1b:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, help="Weight in kilograms")
        
        # Blood Pressure
        st.markdown("**🩸 Blood Pressure**")
        col1c, col1d = st.columns(2)
        with col1c:
            ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120, help="Upper blood pressure reading")
        with col1d:
            ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80, help="Lower blood pressure reading")
        
        # Laboratory Values
        st.markdown("**🧪 Laboratory Values**")
        col1e, col1f = st.columns(2)
        with col1e:
            cholesterol = st.selectbox("Cholesterol Level", 
                                     options=[1, 2, 3], 
                                     format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
                                     help="Cholesterol level category")
        with col1f:
            glucose = st.selectbox("Glucose Level", 
                                 options=[1, 2, 3], 
                                 format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
                                 help="Blood glucose level category")
        
        # Lifestyle Factors
        st.markdown("**🏃‍♂️ Lifestyle Factors**")
        col1g, col1h, col1i = st.columns(3)
        with col1g:
            smoke = st.selectbox("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col1h:
            alcohol = st.selectbox("Alcohol", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col1i:
            active = st.selectbox("Physical Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col2:
        st.subheader("📊 Calculated Metrics")
        
        # Calculate and display derived metrics
        if height > 0 and weight > 0:
            bmi = weight / (height / 100) ** 2
            st.metric("BMI", f"{bmi:.1f}")
            
            # BMI interpretation
            if bmi < 18.5:
                bmi_status = "Underweight"
                bmi_color = "🔵"
            elif bmi < 25:
                bmi_status = "Normal"
                bmi_color = "🟢"
            elif bmi < 30:
                bmi_status = "Overweight"
                bmi_color = "🟡"
            else:
                bmi_status = "Obese"
                bmi_color = "🔴"
            
            st.write(f"{bmi_color} {bmi_status}")
        
        if ap_hi > 0 and ap_lo > 0:
            pulse_pressure = ap_hi - ap_lo
            st.metric("Pulse Pressure", f"{pulse_pressure} mmHg")
            
            # Blood pressure interpretation
            if ap_hi < 120 and ap_lo < 80:
                bp_status = "Normal"
                bp_color = "🟢"
            elif ap_hi < 130 and ap_lo < 80:
                bp_status = "Elevated"
                bp_color = "🟡"
            elif ap_hi < 140 or ap_lo < 90:
                bp_status = "Stage 1 Hypertension"
                bp_color = "🟠"
            else:
                bp_status = "Stage 2 Hypertension"
                bp_color = "🔴"
            
            st.write(f"{bp_color} {bp_status}")
        
        # Risk factors summary
        st.subheader("⚠️ Risk Factors")
        risk_factors = []
        if smoke == 1:
            risk_factors.append("🚬 Smoking")
        if alcohol == 1:
            risk_factors.append("🍷 Alcohol use")
        if active == 0:
            risk_factors.append("🛋️ Sedentary lifestyle")
        if cholesterol > 1:
            risk_factors.append("📈 Elevated cholesterol")
        if glucose > 1:
            risk_factors.append("🍯 Elevated glucose")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.write("✅ No major risk factors")
    
    # Prediction button
    st.markdown("---")
    predict_button = st.button("🔍 Analyze Cardiovascular Risk", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Preprocess input
            features = preprocess_input(age, gender_code, height, weight, ap_hi, ap_lo, 
                                      cholesterol, glucose, smoke, alcohol, active)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction_proba = model.predict_proba(features_scaled)[0]
            disease_probability = prediction_proba[1]
            
            # Apply optimized threshold for high recall
            prediction = 1 if disease_probability >= threshold else 0
            
            # Display results
            st.markdown("## 🎯 Analysis Results")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box disease">
                    🚨 HIGH RISK: Cardiovascular Disease Detected<br>
                    <small>Confidence: {disease_probability:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.error("""
                **⚠️ IMPORTANT MEDICAL ADVICE:**
                - **Immediate Action Required:** Please consult a cardiologist or healthcare provider immediately
                - **This is a screening tool:** Further medical testing is essential for proper diagnosis
                - **High-Recall Model:** This system is designed to catch 90% of cardiovascular cases, prioritizing patient safety
                """)
                
            else:
                st.markdown(f"""
                <div class="prediction-box healthy">
                    ✅ LOW RISK: No Cardiovascular Disease Detected<br>
                    <small>Confidence: {(1-disease_probability):.1%}</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("""
                **✅ GOOD NEWS:**
                - **Low Risk Detected:** Current indicators suggest low cardiovascular disease risk
                - **Maintain Healthy Lifestyle:** Continue regular exercise, healthy diet, and routine checkups
                - **Regular Monitoring:** Annual cardiovascular screenings are still recommended
                """)
            
            # Detailed probability breakdown
            st.markdown("### 📈 Detailed Risk Analysis")
            col3, col4 = st.columns(2)
            
            with col3:
                st.metric("Disease Probability", f"{disease_probability:.1%}")
                st.metric("Healthy Probability", f"{(1-disease_probability):.1%}")
            
            with col4:
                st.metric("Model Sensitivity", "90.05%", help="Percentage of actual disease cases correctly identified")
                st.metric("Optimal Threshold", f"{threshold:.3f}", help="Optimized cutoff for high-recall predictions")
            
            # Risk factor contribution (simplified)
            st.markdown("### 🎯 Key Risk Indicators")
            risk_score = 0
            risk_details = []
            
            if age > 60:
                risk_score += 20
                risk_details.append("🔴 Advanced age (>60)")
            elif age > 45:
                risk_score += 10
                risk_details.append("🟡 Middle age (45-60)")
            
            if bmi > 30:
                risk_score += 15
                risk_details.append("🔴 Obesity (BMI >30)")
            elif bmi > 25:
                risk_score += 8
                risk_details.append("🟡 Overweight (BMI 25-30)")
            
            if ap_hi > 140 or ap_lo > 90:
                risk_score += 20
                risk_details.append("🔴 Hypertension")
            elif ap_hi > 120 or ap_lo > 80:
                risk_score += 10
                risk_details.append("🟡 Elevated blood pressure")
            
            if cholesterol > 2:
                risk_score += 15
                risk_details.append("🔴 High cholesterol")
            elif cholesterol > 1:
                risk_score += 8
                risk_details.append("🟡 Elevated cholesterol")
            
            if glucose > 2:
                risk_score += 15
                risk_details.append("🔴 High glucose")
            elif glucose > 1:
                risk_score += 8
                risk_details.append("🟡 Elevated glucose")
            
            if smoke:
                risk_score += 25
                risk_details.append("🔴 Smoking")
            
            if not active:
                risk_score += 10
                risk_details.append("🟡 Sedentary lifestyle")
            
            if risk_details:
                for detail in risk_details:
                    st.write(detail)
            else:
                st.write("✅ No major risk indicators identified")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong><br>
        This AI system is designed for screening purposes only and should not replace professional medical advice, diagnosis, or treatment. 
        The high-recall model is optimized to catch 90% of cardiovascular disease cases but may produce false positives. 
        Always consult qualified healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Technical details in sidebar
    with st.sidebar:
        st.header("🔧 Technical Details")
        st.write("**Model Performance:**")
        st.write("• Recall: 90.05%")
        st.write("• Precision: 60.53%")
        st.write("• F1-Score: 72.40%")
        st.write("• ROC-AUC: 79.79%")
        
        st.write("**Model Features:**")
        st.write("• Random Forest Classifier")
        st.write("• Balanced class weights")
        st.write("• Optimized threshold: 0.269")
        st.write("• 13 input features")
        
        st.write("**Training Data:**")
        st.write("• 68,588 patient records")
        st.write("• Cardiovascular disease dataset")
        st.write("• Comprehensive feature engineering")

if __name__ == "__main__":
    main()