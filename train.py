import os
import sys
import numpy as np
import pandas as pd
import time

import config
from src.data_loader import BreastCancerDataLoader
from src.preprocessing import DataPreprocessor
from src.classical_models import ClassicalMLPipeline
from src.evaluation import BenchmarkEvaluator
from src.explainability import ModelExplainability
from src.prediction import ModelRegistryManager


def run_training_pipeline():
    """
    Executes the complete Hybrid Quantum-Classical ML Pipeline:
      1. Dataset Ingestion & Target Remapping
      2. Stratified Train/Test Split (80/20)
      3. StandardScaler & 4-Component PCA Feature Reduction
      4. Classical Model Training (SVM, Random Forest, XGBoost)
      5. Quantum Model Training (QSVM, VQC with Qiskit)
      6. Fair Model Comparison & Metrics Evaluation
      7. Confusion Matrix Plotting & ROC Curve Generation
      8. Model Interpretability & Feature Importance
      9. Model Registry Artifact Updates
    """
    print("=" * 70)
    print("   HYBRID QUANTUM-CLASSICAL ML PLATFORM FOR BREAST CANCER DETECTION   ")
    print("=" * 70 + "\n")

    # STAGE 1: Data Ingestion & Validation
    loader = BreastCancerDataLoader()
    df = loader.load_data()
    loader.validate_dataset()

    # STAGE 2: Preprocessing & 4-Component PCA
    preprocessor = DataPreprocessor()
    data = preprocessor.prepare_data(df)
    
    X_train_pca = data["X_train_pca"]
    X_test_pca = data["X_test_pca"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    evaluator = BenchmarkEvaluator()
    registry_manager = ModelRegistryManager()
    roc_data_dict = {}

    # STAGE 3: Classical ML Models
    classical_pipeline = ClassicalMLPipeline()
    classical_models = classical_pipeline.train_all(X_train_pca, y_train)

    for model_name in classical_models.keys():
        y_pred, y_probs = classical_pipeline.predict(model_name, X_test_pca)
        training_time = classical_pipeline.training_times[model_name]
        
        # Record metrics & plot confusion matrix
        metrics = evaluator.evaluate_model(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred,
            y_probs=y_probs,
            training_time=training_time,
            qubits=0
        )
        
        roc_data_dict[model_name] = y_probs
        registry_manager.register_model(model_name, metrics, training_time, qubits=0)

    # STAGE 4: Quantum ML Models (QSVM & VQC on Simulator)
    try:
        from src.quantum_models import QuantumMLPipeline
        qml_pipeline = QuantumMLPipeline()
        
        if qml_pipeline.qiskit_available:
            X_q_train = X_train_pca[:120]
            y_q_train = y_train[:120]
            
            qml_models = qml_pipeline.train_all(X_q_train, y_q_train)

            for model_name in qml_models.keys():
                y_pred, y_probs = qml_pipeline.predict(model_name, X_test_pca)
                training_time = qml_pipeline.training_times[model_name]
                
                metrics = evaluator.evaluate_model(
                    model_name=model_name,
                    y_true=y_test,
                    y_pred=y_pred,
                    y_probs=y_probs,
                    training_time=training_time,
                    qubits=config.N_QUBITS
                )
                
                roc_data_dict[model_name] = y_probs
                registry_manager.register_model(model_name, metrics, training_time, qubits=config.N_QUBITS)
        else:
            print("[INFO] Quantum model training skipped due to Qiskit environment configuration.")
    except Exception as e:
        print(f"[WARNING] Quantum execution notice: {e}")

    # STAGE 5: Save Benchmark Results & Combined Plots
    summary_df = evaluator.save_summary_table()
    evaluator.plot_combined_roc_curves(roc_data_dict, y_test)

    # STAGE 6: Feature Importance Explainability
    explainer = ModelExplainability(feature_names=data["feature_names"])
    if "Random Forest" in classical_models:
        rf_imp = explainer.get_tree_feature_importance(classical_models["Random Forest"], "Random Forest")
        explainer.save_feature_importance(rf_imp)

    # STAGE 7: Criteria-Based Model Selection
    best_model_series = evaluator.select_best_model(summary_df, criterion="Accuracy")
    
    print("\n" + "=" * 70)
    print("                     FINAL BENCHMARK RESULTS                     ")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("-" * 70)
    print(f"Highest-performing model under selected evaluation criterion (Accuracy):")
    print(f" -> {best_model_series['Model']} (Accuracy: {best_model_series['Accuracy'] * 100:.2f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_training_pipeline()
