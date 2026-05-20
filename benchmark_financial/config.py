"""
Configuration isolée du benchmark financier CPB.
Toutes les constantes de ce benchmark sont ici — aucune dépendance sur le config.py racine.
"""
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
ROOT_DIR = BENCHMARK_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# --- Fichiers / répertoires (tous dans benchmark_financial/) ---
DATA_FILE       = BENCHMARK_DIR / "data" / "financial_pii_data.xlsx"
CHROMA_DIR      = str(BENCHMARK_DIR / "chroma_db")
QUERIES_FILE    = BENCHMARK_DIR / "queries.json"
MLFLOW_DIR      = str(BENCHMARK_DIR / "mlruns")
COLLECTION_NAME = "financial_benchmark"

# --- MLflow ---
MLFLOW_EXPERIMENT = "benchmark_financial_cpb"

# --- Distribution des 1000 queries ---
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

# --- Colonnes du xlsx ---
COL_NAME    = "Name"
COL_CC      = "Credit Card"
COL_EMAIL   = "Email"
COL_URL     = "URL"
COL_PHONE   = "Phone"
COL_ADDRESS = "Address"
COL_COMPANY = "Company"
COL_SSN     = "SSN"
COL_TEXT    = "Text"
COL_LABELS  = "True Predictions"
