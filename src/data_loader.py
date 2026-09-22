import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import sys
import os

# Ensure config can be imported from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BreastCancerDataLoader:
    """
    Data Loader module for Breast Cancer Wisconsin Diagnostic Dataset.
    Handles data extraction, label conversion to standard medical decision-support convention,
    and dataset validation checks.
    """

    def __init__(self):
        self.raw_dataset = None
        self.df = None
        self.feature_names = []
        self.target_name = "target"
        self.target_label_map = {1: "Malignant", 0: "Benign"}

    def load_data(self) -> pd.DataFrame:
        """
        Loads dataset from scikit-learn and performs target label remapping.
        
        Note on Label Mapping:
        In sklearn load_breast_cancer: 0 = Malignant, 1 = Benign.
        To comply with clinical risk modeling guidelines (1 = Positive/Malignant, 0 = Negative/Benign),
        we explicitly invert labels:
            1 -> 0 (Benign)
            0 -> 1 (Malignant)
        """
        print("[INFO] Loading Breast Cancer Wisconsin Diagnostic Dataset...")
        dataset = load_breast_cancer()
        self.raw_dataset = dataset
        self.feature_names = list(dataset.feature_names)

        # Create base DataFrame
        df = pd.DataFrame(data=dataset.data, columns=self.feature_names)
        
        # Original sklearn targets: 0=Malignant, 1=Benign
        original_targets = dataset.target
        
        # Remap targets: 0 -> 1 (Malignant), 1 -> 0 (Benign)
        remapped_targets = np.where(original_targets == 0, 1, 0)
        df[self.target_name] = remapped_targets
        
        self.df = df
        print(f"[SUCCESS] Dataset loaded successfully with {df.shape[0]} samples and {df.shape[1] - 1} features.")
        return df

    def validate_dataset(self) -> dict:
        """
        Performs data integrity checks: missing values, duplicates, and class distributions.
        Returns a dictionary summary of validation results.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        print("\n" + "=" * 50)
        print("         DATASET VALIDATION REPORT         ")
        print("=" * 50)

        # 1. Dataset Shape Inspection
        n_rows, n_cols = self.df.shape
        n_features = n_cols - 1

        # 2. Missing Value Check
        missing_count = self.df.isnull().sum().sum()

        # 3. Duplicate Check
        duplicate_count = self.df.duplicated().sum()

        # 4. Class Distribution Inspection
        class_counts = self.df[self.target_name].value_counts().to_dict()
        class_percentages = (self.df[self.target_name].value_counts(normalize=True) * 100).to_dict()

        malignant_count = class_counts.get(1, 0)
        benign_count = class_counts.get(0, 0)

        report = {
            "n_samples": n_rows,
            "n_features": n_features,
            "missing_values": missing_count,
            "duplicate_rows": duplicate_count,
            "malignant_count": malignant_count,
            "benign_count": benign_count,
            "malignant_pct": class_percentages.get(1, 0.0),
            "benign_pct": class_percentages.get(0, 0.0)
        }

        print(f"Total Samples          : {n_rows}")
        print(f"Numerical Features     : {n_features}")
        print(f"Missing Values Count   : {missing_count}")
        print(f"Duplicate Rows Count   : {duplicate_count}")
        print("-" * 50)
        print(f"Class 1 (Malignant)    : {malignant_count} ({report['malignant_pct']:.2f}%)")
        print(f"Class 0 (Benign)       : {benign_count} ({report['benign_pct']:.2f}%)")
        print("=" * 50 + "\n")

        return report

    def get_feature_list(self) -> list:
        """Returns the list of 30 feature names in the dataset."""
        return self.feature_names


if __name__ == "__main__":
    loader = BreastCancerDataLoader()
    df = loader.load_data()
    report = loader.validate_dataset()
