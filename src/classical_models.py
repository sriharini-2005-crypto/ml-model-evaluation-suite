import time
import joblib
import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Ensure config can be imported from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import BreastCancerDataLoader
from src.preprocessing import DataPreprocessor


class ClassicalMLPipeline:
    """
    Classical Machine Learning Training and Evaluation Pipeline.
    Implements:
      1. Support Vector Classifier (RBF Kernel, probability=True)
      2. Random Forest Classifier (200 trees, balanced class weights)
      3. XGBoost Classifier (200 trees, max depth 4, learning rate 0.05)
    """

    def __init__(self, random_state=config.RANDOM_STATE):
        self.random_state = random_state
        self.models = {
            "SVM": SVC(
                kernel="rbf",
                probability=True,
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight="balanced"
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state,
                eval_metric="logloss"
            )
        }
        self.trained_models = {}
        self.training_times = {}

    def train_all(self, X_train_pca: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Trains all 3 classical models on the PCA-transformed feature set.
        Measures precise training execution time for each model.
        """
        print("\n" + "=" * 50)
        print("          TRAINING CLASSICAL ML MODELS          ")
        print("=" * 50)

        for model_name, model in self.models.items():
            print(f"[INFO] Training {model_name}...")
            start_time = time.time()
            model.fit(X_train_pca, y_train)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            self.training_times[model_name] = elapsed_time
            self.trained_models[model_name] = model

            # Save trained model artifact
            file_name = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            save_path = os.path.join(config.MODELS_DIR, file_name)
            joblib.dump(model, save_path)
            
            print(f"[SUCCESS] {model_name} trained in {elapsed_time:.4f}s -> Saved: {save_path}")

        print("=" * 50 + "\n")
        return self.trained_models

    def predict(self, model_name: str, X_test_pca: np.ndarray) -> tuple:
        """
        Generates class predictions and prediction probabilities for a given model.
        Returns (y_pred, y_probs).
        """
        if model_name not in self.trained_models:
            raise KeyError(f"Model {model_name} not trained yet. Call train_all first.")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test_pca)
        
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test_pca)[:, 1]
        elif hasattr(model, "decision_function"):
            # Fallback for decision boundary scoring if probability is disabled
            y_probs = model.decision_function(X_test_pca)
        else:
            y_probs = None

        return y_pred, y_probs


if __name__ == "__main__":
    loader = BreastCancerDataLoader()
    df = loader.load_data()
    
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)
    
    classical_pipeline = ClassicalMLPipeline()
    trained_models = classical_pipeline.train_all(
        data_dict["X_train_pca"],
        data_dict["y_train"]
    )
    
    # Test sample inference
    for name in trained_models.keys():
        y_pred, y_probs = classical_pipeline.predict(name, data_dict["X_test_pca"])
        print(f"[VERIFICATION] {name} test predictions generated ({len(y_pred)} samples).")
