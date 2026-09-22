import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import config
from src.data_loader import BreastCancerDataLoader
from src.preprocessing import DataPreprocessor
from src.classical_models import ClassicalMLPipeline
from src.evaluation import BenchmarkEvaluator
from src.explainability import ModelExplainability
from src.prediction import PatientPredictor, ModelRegistryManager

app = FastAPI(
    title="Hybrid QML Disease Detection API",
    description="Backend API for Breast Cancer Risk Classification comparing Classical and Quantum Machine Learning",
    version="1.0.0"
)

# Enable CORS for React Frontend (typically running on localhost:3000 or localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static results directory (confusion matrices & ROC curves)
os.makedirs(config.RESULTS_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=config.RESULTS_DIR), name="results")


class PatientPredictionRequest(BaseModel):
    features: List[float]
    selected_model: str = "Random Forest"


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "disease": "Breast Cancer Risk Classification",
        "architecture": "Hybrid Quantum-Classical ML",
        "qiskit_qubits": config.N_QUBITS
    }


@app.get("/api/dataset/default")
def get_default_dataset_info():
    """Loads default Wisconsin Breast Cancer dataset stats and feature names."""
    try:
        loader = BreastCancerDataLoader()
        df = loader.load_data()
        report = loader.validate_dataset()
        feature_names = loader.get_feature_list()
        sample_row = df.drop(columns=["target"]).mean().to_dict()

        return {
            "dataset_name": "Breast Cancer Wisconsin Diagnostic",
            "samples": report["n_samples"],
            "features_count": report["n_features"],
            "feature_names": feature_names,
            "class_distribution": {
                "Malignant (1)": report["malignant_count"],
                "Benign (0)": report["benign_count"]
            },
            "sample_defaults": sample_row,
            "preview": df.head(10).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Allows doctors/users to upload a custom CSV dataset."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    upload_path = os.path.join(config.DATA_DIR, "uploaded_dataset.csv")
    try:
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)

        df = pd.read_csv(upload_path)
        if "target" not in df.columns and "diagnosis" not in df.columns:
            # Check last column as target if target column name differs
            target_col = df.columns[-1]
            df.rename(columns={target_col: "target"}, inplace=True)

        n_rows, n_cols = df.shape
        missing_count = int(df.isnull().sum().sum())
        feature_names = [c for c in df.columns if c != "target"]

        return {
            "filename": file.filename,
            "samples": n_rows,
            "features_count": len(feature_names),
            "feature_names": feature_names,
            "missing_values": missing_count,
            "status": "Dataset uploaded and validated successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


@app.post("/api/train")
def train_and_benchmark():
    """Executes the full preprocessing, training, benchmarking, and selection pipeline."""
    try:
        # 1. Load Data
        upload_path = os.path.join(config.DATA_DIR, "uploaded_dataset.csv")
        if os.path.exists(upload_path):
            df = pd.read_csv(upload_path)
        else:
            loader = BreastCancerDataLoader()
            df = loader.load_data()

        # 2. Preprocess
        preprocessor = DataPreprocessor()
        data = preprocessor.prepare_data(df)

        evaluator = BenchmarkEvaluator()
        registry_manager = ModelRegistryManager()
        roc_data_dict = {}

        # 3. Train Classical Models
        classical_pipeline = ClassicalMLPipeline()
        classical_models = classical_pipeline.train_all(data["X_train_pca"], data["y_train"])

        benchmark_results = []
        for model_name in classical_models.keys():
            y_pred, y_probs = classical_pipeline.predict(model_name, data["X_test_pca"])
            t_time = classical_pipeline.training_times[model_name]
            
            metrics = evaluator.evaluate_model(
                model_name=model_name,
                y_true=data["y_test"],
                y_pred=y_pred,
                y_probs=y_probs,
                training_time=t_time,
                qubits=0
            )
            roc_data_dict[model_name] = y_probs
            registry_manager.register_model(model_name, metrics, t_time, qubits=0)
            benchmark_results.append(metrics)

        # 4. Quantum Simulation Execution (QSVM & VQC on AerSimulator / Sampler)
        try:
            from src.quantum_models import QuantumMLPipeline
            qml_pipeline = QuantumMLPipeline()
            if qml_pipeline.qiskit_available:
                # Sub-sample 120 training samples for quantum circuit simulator responsiveness
                X_q_train = data["X_train_pca"][:120]
                y_q_train = data["y_train"][:120]
                
                qml_models = qml_pipeline.train_all(X_q_train, y_q_train)
                for model_name in qml_models.keys():
                    y_pred, y_probs = qml_pipeline.predict(model_name, data["X_test_pca"])
                    t_time = qml_pipeline.training_times[model_name]
                    
                    metrics = evaluator.evaluate_model(
                        model_name=model_name,
                        y_true=data["y_test"],
                        y_pred=y_pred,
                        y_probs=y_probs,
                        training_time=t_time,
                        qubits=config.N_QUBITS
                    )
                    roc_data_dict[model_name] = y_probs
                    registry_manager.register_model(model_name, metrics, t_time, qubits=config.N_QUBITS)
                    benchmark_results.append(metrics)
        except Exception as q_err:
            print(f"[WARNING] Quantum Simulation notice: {q_err}")

        # Save Benchmark artifacts
        summary_df = evaluator.save_summary_table()
        evaluator.plot_combined_roc_curves(roc_data_dict, data["y_test"])

        # Explainability
        explainer = ModelExplainability(feature_names=data["feature_names"])
        if "Random Forest" in classical_models:
            rf_imp = explainer.get_tree_feature_importance(classical_models["Random Forest"], "Random Forest")
            explainer.save_feature_importance(rf_imp)

        best_row = evaluator.select_best_model(summary_df, criterion="Accuracy")
        records = summary_df.to_dict(orient="records")
        clean_records = []
        for r in records:
            clean_r = {}
            for k, v in r.items():
                if pd.isna(v):
                    clean_r[k] = None
                else:
                    clean_r[k] = v
            clean_records.append(clean_r)

        return {
            "status": "Training and benchmarking complete.",
            "metrics": clean_records,
            "best_model": {
                "name": best_row["Model"],
                "accuracy": float(best_row["Accuracy"]),
                "sensitivity": float(best_row["Sensitivity"]),
                "specificity": float(best_row["Specificity"]),
                "f1": float(best_row["F1"]),
                "roc_auc": float(best_row["ROC-AUC"]) if (pd.notna(best_row["ROC-AUC"])) else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark")
def get_benchmark_results():
    """Returns saved comparison metrics CSV data."""
    if not os.path.exists(config.METRICS_CSV_PATH):
        raise HTTPException(status_code=404, detail="Benchmark metrics not found. Run training first.")
    
    df = pd.read_csv(config.METRICS_CSV_PATH)
    records = df.to_dict(orient="records")
    clean_records = []
    for r in records:
        clean_r = {}
        for k, v in r.items():
            if pd.isna(v):
                clean_r[k] = None
            else:
                clean_r[k] = v
        clean_records.append(clean_r)

    return {
        "metrics": clean_records,
        "roc_curve_url": "/results/roc_curves.png"
    }


@app.post("/api/predict")
def predict_patient_sample(request: PatientPredictionRequest):
    """Generates patient risk classification using the selected model."""
    try:
        model_name = request.selected_model
        file_name = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        model_path = os.path.join(config.MODELS_DIR, file_name)

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Trained model '{model_name}' not found.")

        try:
            model = joblib.load(model_path)
        except Exception:
            import dill
            with open(model_path, "rb") as f:
                model = dill.load(f)

        predictor = PatientPredictor()
        result = predictor.predict_sample(np.array(request.features), model, model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/registry")
def get_model_registry():
    """Returns model registry entries."""
    manager = ModelRegistryManager()
    return manager.get_all_entries()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
