import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# Ensure config can be imported from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BenchmarkEvaluator:
    """
    Evaluator module for calculating comprehensive model-level metrics,
    generating confusion matrices, plotting ROC curves, and performing criteria-based model selection.
    """

    def __init__(self, results_dir=config.RESULTS_DIR):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.metrics_list = []

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None, training_time: float = 0.0, qubits: int = 0) -> dict:
        """
        Computes Accuracy, Sensitivity, Specificity, Precision, F1-Score, and ROC-AUC.
        """
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # Confusion matrix for Specificity: TN / (TN + FP)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # ROC-AUC calculation
        auc_score = np.nan
        if y_probs is not None:
            try:
                auc_score = roc_auc_score(y_true, y_probs)
            except Exception:
                auc_score = np.nan

        return {
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": prec,
            "F1": f1,
            "ROC-AUC": auc_score,
            "Training Time (s)": training_time,
            "Qubits": qubits
        }

    def evaluate_model(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None, training_time: float = 0.0, qubits: int = 0):
        """Calculates metrics, plots confusion matrix, and stores summary."""
        metrics = self.calculate_metrics(y_true, y_pred, y_probs, training_time, qubits)
        metrics["Model"] = model_name
        self.metrics_list.append(metrics)

        # Plot individual Confusion Matrix
        self.plot_confusion_matrix(model_name, y_true, y_pred)
        return metrics

    def plot_confusion_matrix(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray):
        """Generates and saves styled confusion matrix figure."""
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Malignant (1)", "Benign (0)"],
            yticklabels=["Malignant (1)", "Benign (0)"]
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("Actual Target")
        plt.xlabel("Predicted Target")
        plt.tight_layout()

        file_name = f"confusion_{model_name.lower().replace(' ', '_')}.png"
        file_path = os.path.join(self.results_dir, file_name)
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved Confusion Matrix for {model_name} -> {file_path}")

    def plot_combined_roc_curves(self, roc_data_dict: dict, y_true: np.ndarray):
        """
        Generates and saves combined ROC curves plot for models with valid prediction probabilities.
        roc_data_dict format: { 'ModelName': y_probs }
        """
        plt.figure(figsize=(8, 6))
        
        for model_name, y_probs in roc_data_dict.items():
            if y_probs is not None and not np.isnan(y_probs).all():
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=1)
                    auc_val = roc_auc_score(y_true, y_probs)
                    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})")
                except Exception:
                    continue

        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.500)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curves Comparison (Classical vs Quantum Models)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        file_path = os.path.join(self.results_dir, "roc_curves.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved Combined ROC Curves -> {file_path}")

    def save_summary_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.metrics_list)
        # Reorder columns
        cols = ["Model", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC-AUC", "Training Time (s)", "Qubits"]
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]

        save_path = config.METRICS_CSV_PATH
        df.to_csv(save_path, index=False)
        print(f"\n[SUCCESS] Model Benchmark Summary saved -> {save_path}")
        return df

    @staticmethod
    def select_best_model(df: pd.DataFrame, criterion: str = "Accuracy") -> pd.Series:
        """
        Selects highest-performing model based on the chosen evaluation metric.
        Wording: 'Highest-performing model under the selected evaluation criterion.'
        """
        if criterion not in df.columns:
            raise KeyError(f"Criterion '{criterion}' not found in benchmark table.")
        best_idx = df[criterion].idxmax()
        best_row = df.loc[best_idx]
        return best_row
