import os

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Ensure required directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, SRC_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_PCA_COMPONENTS = 4

# Target class labeling (scikit-learn breast_cancer defaults: 0 = malignant, 1 = benign)
# We map explicitly: 1 = Malignant, 0 = Benign for medical decision alignment
TARGET_MAP = {
    0: "Malignant",
    1: "Benign"
}

# Quantum Configuration
N_QUBITS = N_PCA_COMPONENTS
QUANTUM_REPETITIONS = 2
VQC_MAX_ITER = 100

# Saved artifact file paths
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
PCA_PATH = os.path.join(MODELS_DIR, "pca.pkl")
METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")
MODEL_REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")
FEATURE_IMPORTANCE_PATH = os.path.join(RESULTS_DIR, "feature_importance.csv")
