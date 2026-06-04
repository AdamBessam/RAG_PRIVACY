"""
Configuration isolée du benchmark ASQ-PHI CPB.
"""
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
ROOT_DIR      = BENCHMARK_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# --- Source des données ---
DATA_URL  = "https://raw.githubusercontent.com/JamesWeatherhead/asq-phi/main/data/synthetic_clinical_queries.txt"
DATA_FILE = BENCHMARK_DIR / "data" / "synthetic_clinical_queries.txt"

# --- Fichiers / répertoires ---
CHROMA_DIR      = str(BENCHMARK_DIR / "chroma_db")
QUERIES_FILE    = BENCHMARK_DIR / "queries.json"
MLFLOW_DIR      = str(BENCHMARK_DIR / "mlruns")
COLLECTION_NAME = "asq_phi_benchmark"

# --- MLflow ---
MLFLOW_EXPERIMENT = "benchmark_asq_phi_cpb"

# --- Distribution des queries ---
N_NORMAL    = 400
N_IKEA      = 200
N_INJECTION = 200
N_DGEA      = 100
N_MIA       = 100
N_TOTAL     = N_NORMAL + N_IKEA + N_INJECTION + N_DGEA + N_MIA  # 1000

# --- RAG ---
TOP_K = 5

# --- Reproductibilité ---
RANDOM_SEED = 42
