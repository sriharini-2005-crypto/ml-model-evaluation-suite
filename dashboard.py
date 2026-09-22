import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from PIL import Image

# Ensure config can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import BreastCancerDataLoader
from src.preprocessing import DataPreprocessor
from src.prediction import PatientPredictor, ModelRegistryManager
from src.evaluation import BenchmarkEvaluator


# Set Streamlit Page Config
st.set_page_config(
    page_title="Hybrid QML Disease Detection Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 4px;
        color: #991B1B;
        font-size: 0.95rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title("🧬 Hybrid QML Platform")
st.sidebar.markdown("**Prototype**")
st.sidebar.markdown("---")

module = st.sidebar.radio(
    "Select Module:",
    [
        "Overview",
        "Dataset Inspection",
        "Model Benchmark",
        "Model Selection",
        "Patient Risk Prediction",
        "Explainability",
        "Model Registry"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("🔒 *Research Prototype — Not a Clinical Diagnostic Tool*")


# Header Banner
st.markdown('<div class="main-title">Hybrid Quantum-Classical ML Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Early Disease Detection Benchmark: Breast Cancer Risk Classification</div>', unsafe_allow_html=True)


# MODULE 1: OVERVIEW
if module == "Overview":
    st.header("📌 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Target Disease", value="Breast Cancer")
    with col2:
        st.metric(label="Total Samples", value="569 Patients")
    with col3:
        st.metric(label="Original Features", value="30 Numerical")
    with col4:
        st.metric(label="PCA Components / Qubits", value="4 Qubits")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Classical ML Models")
        st.markdown("""
        1. **SVM (Support Vector Machine)** - RBF Kernel
        2. **Random Forest Classifier** - 200 Trees
        3. **XGBoost Classifier** - Gradient Boosted Trees
        """)

    with col_b:
        st.subheader("Quantum ML Models (Qiskit)")
        st.markdown("""
        4. **QSVM (Quantum Kernel SVM)** - 4 Qubits, ZZFeatureMap
        5. **VQC (Variational Quantum Classifier)** - 4 Qubits, RealAmplitudes Ansatz
        """)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ IMPORTANT MEDICAL RESEARCH DISCLAIMER</strong><br>
        This software is a research decision-support prototype built to compare classical and quantum machine learning algorithms under predefined evaluation criteria. It is <strong>NOT</strong> a clinical diagnostic system and should never replace qualified healthcare professionals.
    </div>
    """, unsafe_allow_html=True)


# MODULE 2: DATASET INSPECTION
elif module == "Dataset Inspection":
    st.header("📊 Dataset Inspection & PCA Reduction")
    
    loader = BreastCancerDataLoader()
    df = loader.load_data()
    
    tab1, tab2, tab3 = st.tabs(["Dataset Preview", "Class Distribution", "PCA Reduction"])
    
    with tab1:
        st.subheader("Breast Cancer Wisconsin Diagnostic Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")

    with tab2:
        st.subheader("Target Class Distribution")
        class_counts = df["target"].value_counts().rename({1: "Malignant (1)", 0: "Benign (0)"})
        st.bar_chart(class_counts)
        st.write("Target Mapping: `1 = Malignant`, `0 = Benign`")

    with tab3:
        st.subheader("StandardScaler & 4-Component PCA")
        st.markdown("""
        * **Original Features:** 30 numerical cell nucleus measurements
        * **Reduced Dimension:** 4 Principal Components (Resource constraint for 4-qubit quantum feature map)
        * **Cumulative Explained Variance:** ~79.32% of total dataset variance retained
        """)


# MODULE 3: MODEL BENCHMARK
elif module == "Model Benchmark":
    st.header("📈 Common Model Benchmarking")
    
    if os.path.exists(config.METRICS_CSV_PATH):
        df_metrics = pd.read_csv(config.METRICS_CSV_PATH)
        st.subheader("Fair Benchmark Table (Same PCA Data & Train/Test Split)")
        st.dataframe(df_metrics, use_container_width=True)

        st.subheader("Metric Comparison Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Model Accuracy**")
            st.bar_chart(df_metrics.set_index("Model")["Accuracy"])
        with col2:
            st.write("**Model Sensitivity (Recall)**")
            st.bar_chart(df_metrics.set_index("Model")["Sensitivity"])

        st.subheader("Confusion Matrices & ROC Curves")
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Combined ROC Curves**")
            roc_path = os.path.join(config.RESULTS_DIR, "roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)
        with col4:
            st.write("**Sample Confusion Matrix (SVM)**")
            cm_path = os.path.join(config.RESULTS_DIR, "confusion_svm.png")
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
    else:
        st.warning("Benchmark results not found. Please run `python train.py` to train models.")


# MODULE 4: MODEL SELECTION
elif module == "Model Selection":
    st.header("🎯 Criteria-Based Model Selection")
    
    if os.path.exists(config.METRICS_CSV_PATH):
        df_metrics = pd.read_csv(config.METRICS_CSV_PATH)
        
        criterion = st.selectbox(
            "Select Primary Evaluation Criterion:",
            ["Accuracy", "Sensitivity", "Specificity", "F1", "ROC-AUC", "Training Time (s)"]
        )
        
        if criterion == "Training Time (s)":
            best_idx = df_metrics[criterion].idxmin()
        else:
            best_idx = df_metrics[criterion].idxmax()
            
        best_row = df_metrics.loc[best_idx]
        
        st.success(f"**Highest-performing model under selected evaluation criterion ({criterion}):**")
        st.title(f"🏆 {best_row['Model']}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{best_row['Accuracy']*100:.2f}%")
        col2.metric("Sensitivity", f"{best_row['Sensitivity']*100:.2f}%")
        col3.metric("Specificity", f"{best_row['Specificity']*100:.2f}%")
        col4.metric("F1-Score", f"{best_row['F1']*100:.2f}%")
    else:
        st.warning("Benchmark data not found. Please run `python train.py` first.")


# MODULE 5: PATIENT RISK PREDICTION
elif module == "Patient Risk Prediction":
    st.header("🔮 Patient Sample Risk Prediction")
    st.write("Enter patient biomedical measurements to generate prediction using pre-fitted transformers.")

    # Load baseline dataset sample values for slider defaults
    loader = BreastCancerDataLoader()
    df_raw = loader.load_data()
    feature_names = loader.get_feature_list()
    mean_vals = df_raw[feature_names].mean().to_dict()

    st.subheader("Input 30 Cell Measurement Parameters")
    
    # Organize 30 inputs in 3 columns
    input_vals = []
    cols = st.columns(3)
    for idx, feature in enumerate(feature_names):
        col_idx = idx % 3
        val = cols[col_idx].number_input(
            f"{feature}",
            value=float(mean_vals[feature]),
            key=f"feat_{idx}"
        )
        input_vals.append(val)

    # Select Trained Model
    model_choice = st.selectbox("Select Model for Prediction:", ["SVM", "Random Forest", "XGBoost", "QSVM", "VQC"])

    if st.button("RUN PREDICTION", type="primary"):
        try:
            # Map choice to saved model filename
            file_name = f"{model_choice.lower().replace(' ', '_')}_model.pkl"
            model_path = os.path.join(config.MODELS_DIR, file_name)
            
            if not os.path.exists(model_path):
                st.error(f"Model file {model_path} not found. Train models first using `python train.py`.")
            else:
                model = joblib.load(model_path)
                predictor = PatientPredictor()
                result = predictor.predict_sample(input_vals, model, model_choice)
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    if result["predicted_label"] == "Malignant":
                        st.error(f"Predicted Class: **{result['predicted_label']}**")
                    else:
                        st.success(f"Predicted Class: **{result['predicted_label']}**")
                
                with res_col2:
                    if result["confidence_percentage"] is not None:
                        st.metric("Prediction Probability", f"{result['confidence_percentage']:.2f}%")
                    else:
                        st.metric("Prediction Probability", "N/A (Decision Boundary)")
                
                with res_col3:
                    st.metric("Selected Model", result["selected_model"])

                st.markdown("""
                <div class="disclaimer-box">
                    <strong>Research Disclaimer:</strong> This is a research prototype and is not a clinical diagnosis.
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# MODULE 6: EXPLAINABILITY
elif module == "Explainability":
    st.header("💡 Model Explainability & Feature Importance")
    
    if os.path.exists(config.FEATURE_IMPORTANCE_PATH):
        df_imp = pd.read_csv(config.FEATURE_IMPORTANCE_PATH)
        st.subheader("Top Influential Features (Random Forest)")
        st.bar_chart(df_imp.set_index("Feature")["Importance"].head(10))
        st.dataframe(df_imp, use_container_width=True)
        
        st.info(
            "Explanation Statement: The features listed above had a high mathematical influence on the model prediction. "
            "This does NOT imply a direct biological cause of disease."
        )
    else:
        st.warning("Feature importance file not found. Run `python train.py` to generate interpretability tables.")


# MODULE 7: MODEL REGISTRY
elif module == "Model Registry":
    st.header("🗂️ Disease-Specific Model Registry")
    
    if os.path.exists(config.MODEL_REGISTRY_PATH):
        with open(config.MODEL_REGISTRY_PATH, "r") as f:
            registry_data = json.load(f)
        
        df_reg = pd.DataFrame(registry_data)
        st.dataframe(df_reg, use_container_width=True)
    else:
        st.warning("Model registry is empty or missing. Run `python train.py` to register trained models.")
