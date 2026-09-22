import time
import os
import sys
import numpy as np
import joblib
import dill

# Ensure config can be imported from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import BreastCancerDataLoader
from src.preprocessing import DataPreprocessor

# Qiskit 1.x & Qiskit Machine Learning imports
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, VQC

try:
    from qiskit.primitives import StatevectorSampler as Sampler
except ImportError:
    from qiskit.primitives import Sampler


class QuantumMLPipeline:
    """
    Quantum Machine Learning Simulation Pipeline.
    Runs 4-qubit QSVM and VQC models locally using Qiskit Aer / Sampler primitives.
    Corresponding to 4 PCA-reduced features.
    """

    def __init__(self, n_qubits=config.N_QUBITS, random_state=config.RANDOM_STATE):
        self.n_qubits = n_qubits
        self.random_state = random_state
        self.trained_models = {}
        self.training_times = {}

        print(f"[INFO] Initializing Quantum Simulator Environment ({self.n_qubits} Qubits)...")
        
        # 1. 4-Qubit Quantum Feature Map
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=2,
            entanglement="linear"
        )

        # 2. Quantum Kernel for QSVM
        self.quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        self.qsvm_model = QSVC(quantum_kernel=self.quantum_kernel)

        # 3. Parameterized Ansatz & Sampler for VQC
        self.ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=2,
            entanglement="linear"
        )
        self.optimizer = COBYLA(maxiter=50)
        
        try:
            self.sampler = Sampler()
            self.vqc_model = VQC(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=self.optimizer,
                sampler=self.sampler
            )
        except Exception:
            self.vqc_model = VQC(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=self.optimizer
            )

        self.qiskit_available = True

    def train_qsvm(self, X_train_pca: np.ndarray, y_train: np.ndarray):
        """Trains 4-Qubit Quantum Support Vector Classifier (QSVM) on simulator."""
        print(f"\n[INFO] Simulating Quantum Model 1: QSVM ({self.n_qubits} Qubits, ZZFeatureMap)...")
        start_time = time.time()
        
        # Train QSVC
        self.qsvm_model.fit(X_train_pca, y_train)
        end_time = time.time()

        elapsed = end_time - start_time
        self.training_times["QSVM"] = elapsed
        self.trained_models["QSVM"] = self.qsvm_model

        save_path = os.path.join(config.MODELS_DIR, "qsvm_model.pkl")
        with open(save_path, "wb") as f:
            dill.dump(self.qsvm_model, f)
        print(f"[SUCCESS] QSVM Quantum Simulation completed in {elapsed:.2f}s -> Saved: {save_path}")
        return self.qsvm_model

    def train_vqc(self, X_train_pca: np.ndarray, y_train: np.ndarray):
        """Trains 4-Qubit Variational Quantum Classifier (VQC) on simulator."""
        print(f"\n[INFO] Simulating Quantum Model 2: VQC ({self.n_qubits} Qubits, RealAmplitudes Ansatz)...")
        start_time = time.time()

        if y_train.ndim == 1:
            num_classes = len(np.unique(y_train))
            y_train_encoded = np.eye(num_classes)[y_train]
        else:
            y_train_encoded = y_train

        # Fit VQC
        self.vqc_model.fit(X_train_pca, y_train_encoded)
        end_time = time.time()

        elapsed = end_time - start_time
        self.training_times["VQC"] = elapsed
        self.trained_models["VQC"] = self.vqc_model

        save_path = os.path.join(config.MODELS_DIR, "vqc_model.pkl")
        with open(save_path, "wb") as f:
            dill.dump(self.vqc_model, f)
        print(f"[SUCCESS] VQC Quantum Simulation completed in {elapsed:.2f}s -> Saved: {save_path}")
        return self.vqc_model

    def train_all(self, X_train_pca: np.ndarray, y_train: np.ndarray) -> dict:
        """Trains both QSVM and VQC quantum models sequentially."""
        print("\n" + "=" * 50)
        print("     STARTING QUANTUM SIMULATION PIPELINE (4 QUBITS)     ")
        print("=" * 50)

        self.train_qsvm(X_train_pca, y_train)
        self.train_vqc(X_train_pca, y_train)

        print("=" * 50 + "\n")
        return self.trained_models

    def predict(self, model_name: str, X_test_pca: np.ndarray) -> tuple:
        """Generates quantum predictions and confidence probabilities."""
        if model_name not in self.trained_models:
            raise KeyError(f"Quantum model {model_name} not trained yet.")

        model = self.trained_models[model_name]
        raw_pred = model.predict(X_test_pca)

        if raw_pred.ndim > 1:
            y_pred = np.argmax(raw_pred, axis=1)
            y_probs = raw_pred[:, 1]
        else:
            y_pred = raw_pred
            y_probs = None

        if hasattr(model, "predict_proba"):
            try:
                y_probs = model.predict_proba(X_test_pca)[:, 1]
            except Exception:
                pass

        return y_pred, y_probs


if __name__ == "__main__":
    loader = BreastCancerDataLoader()
    df = loader.load_data()

    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    # Subset dataset to 100 samples for fast simulation validation
    X_train_sub = data_dict["X_train_pca"][:100]
    y_train_sub = data_dict["y_train"][:100]

    qml_pipeline = QuantumMLPipeline()
    qml_pipeline.train_all(X_train_sub, y_train_sub)
