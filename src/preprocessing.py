import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import sys
import os

# Ensure config can be imported from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import BreastCancerDataLoader


class DataPreprocessor:
    """
    Data Preprocessing and Feature Reduction pipeline.
    Ensures strict separation of train/test data to prevent data leakage:
      - Scaler is fitted ONLY on training set.
      - PCA is fitted ONLY on training set.
    """

    def __init__(self, n_components=config.N_PCA_COMPONENTS, random_state=config.RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        
        self.feature_names = []
        self.explained_variance_ratio_ = None
        self.total_explained_variance_ = 0.0

    def prepare_data(self, df: pd.DataFrame, target_col: str = "target"):
        """
        Splits data, fits scaler & PCA on train set, transforms test set, and saves fitted transformers.
        """
        print("\n" + "=" * 50)
        print("     DATA PREPROCESSING & PCA FEATURE REDUCTION     ")
        print("=" * 50)

        # Separate features and target label
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        self.feature_names = list(X.columns)

        # 1. Stratified Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            stratify=y,
            random_state=self.random_state
        )

        print(f"[INFO] Train/Test Split Complete (Test Size = {config.TEST_SIZE * 100:.0f}%):")
        print(f"       Training Samples : {X_train.shape[0]}")
        print(f"       Testing Samples  : {X_test.shape[0]}")

        # 2. Standardization (Fit ONLY on Training Data)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save fitted StandardScaler artifact
        joblib.dump(self.scaler, config.SCALER_PATH)
        print(f"[SUCCESS] Saved StandardScaler artifact -> {config.SCALER_PATH}")

        # 3. Principal Component Analysis (Fit ONLY on Training Data)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        # Save fitted PCA artifact
        joblib.dump(self.pca, config.PCA_PATH)
        print(f"[SUCCESS] Saved PCA artifact -> {config.PCA_PATH}")

        # Record PCA statistics
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.total_explained_variance_ = float(np.sum(self.explained_variance_ratio_))

        print("-" * 50)
        print(f"Original Feature Count : {X.shape[1]}")
        print(f"Reduced PCA Component  : {self.n_components} components")
        for i, ratio in enumerate(self.explained_variance_ratio_):
            print(f"  - PCA Component {i+1} Explained Variance : {ratio * 100:.2f}%")
        print(f"Total Cumulative Explained Variance       : {self.total_explained_variance_ * 100:.2f}%")
        print("=" * 50 + "\n")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "X_train_pca": X_train_pca,
            "X_test_pca": X_test_pca,
            "feature_names": self.feature_names
        }

    @staticmethod
    def load_preprocessors():
        """Helper to load pre-fitted scaler and PCA for single sample prediction."""
        if not os.path.exists(config.SCALER_PATH) or not os.path.exists(config.PCA_PATH):
            raise FileNotFoundError("Scaler or PCA model files not found. Run training pipeline first.")
        
        scaler = joblib.load(config.SCALER_PATH)
        pca = joblib.load(config.PCA_PATH)
        return scaler, pca


if __name__ == "__main__":
    loader = BreastCancerDataLoader()
    df = loader.load_data()
    
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)
