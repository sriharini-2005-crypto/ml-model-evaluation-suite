import os
import sys
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import joblib

# Ensure config can be imported from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ModelExplainability:
    """
    Explainability module for interpretability.
    Provides feature importances for classical tree-based models (RF, XGBoost)
    and permutation importance for model-agnostic inspection.
    """

    def __init__(self, feature_names: list):
        self.feature_names = feature_names

    def get_tree_feature_importance(self, model, model_name: str) -> pd.DataFrame:
        """Extracts native Gini / Gain feature importances from tree-based models."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            n_imp = len(importances)
            
            if len(self.feature_names) == n_imp:
                names = self.feature_names
            else:
                names = [f"PCA Component {i+1}" for i in range(n_imp)]
                
            df_imp = pd.DataFrame({
                "Feature": names,
                "Importance": importances,
                "Model": model_name
            }).sort_values(by="Importance", ascending=False)
            return df_imp
        return pd.DataFrame()

    def compute_permutation_importance(self, model, X_test, y_test, model_name: str) -> pd.DataFrame:
        """Computes model-agnostic permutation feature importance on test set."""
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1
        )
        df_perm = pd.DataFrame({
            "Feature": self.feature_names,
            "Permutation_Importance_Mean": result.importances_mean,
            "Permutation_Importance_Std": result.importances_std,
            "Model": model_name
        }).sort_values(by="Permutation_Importance_Mean", ascending=False)
        return df_perm

    def save_feature_importance(self, df_imp: pd.DataFrame, save_path: str = config.FEATURE_IMPORTANCE_PATH):
        """Saves feature importance table to CSV."""
        df_imp.to_csv(save_path, index=False)
        print(f"[SUCCESS] Feature Importance saved -> {save_path}")

    @staticmethod
    def get_explanation_disclaimer() -> str:
        """Returns research explanation statement."""
        return (
            "Feature Importance Explanation:\n"
            "The features listed above had a high mathematical influence on the model prediction.\n"
            "This does NOT imply a direct biological cause of disease. It indicates statistical correlation "
            "and model decision contribution."
        )
