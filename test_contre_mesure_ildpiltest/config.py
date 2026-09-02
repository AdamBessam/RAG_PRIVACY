"""
Configuration isolée — benchmark contre-mesure CPB sur ildpil/text-anonymization-benchmark (split test).
Aucune dépendance sur le config.py racine.
"""
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
ROOT_DIR      = BENCHMARK_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# --- Dataset ---
DATASET_NAME  = "ildpil/text-anonymization-benchmark"
DATASET_SPLIT = "test"   # split vierge (jamais indexé dans les autres benchmarks)

# --- Fichiers / répertoires (tous dans test_contre_mesure_ildpiltest/) ---
CHROMA_DIR      = str(BENCHMARK_DIR / "chroma_db")
QUERIES_FILE    = BENCHMARK_DIR / "queries.json"
QUERIES_CSV     = BENCHMARK_DIR / "queries.csv"
RESULTS_CSV     = BENCHMARK_DIR / "benchmark_results.csv"
MLFLOW_DIR      = str(BENCHMARK_DIR / "mlruns")
COLLECTION_NAME = "ildpil_test_benchmark"

# --- MLflow ---
MLFLOW_EXPERIMENT = "cpb_ildpil_test"

# --- Distribution des 1000 queries ---
N_NORMAL    = 300   # questions légitimes sur le contenu
N_DIRECT    = 300   # extraction ciblée de PII (IKEA-style)
N_INJECTION = 200   # prompt injection
N_DGEA      = 100   # jailbreak autoritaire (DGEA-style)
N_MIA       = 100   # membership inference
N_TOTAL     = N_NORMAL + N_DIRECT + N_INJECTION + N_DGEA + N_MIA  # 1000

# --- RAG ---
TOP_K       = 5
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50

# --- Embedding ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLMs ---
OLLAMA_BASE_URL = "http://localhost:11434"
LLAMA_MODEL     = "llama3.1:8b"
GPT4O_MINI_MODEL = "gpt-4o-mini"

# --- Génération ---
MAX_TOKENS  = 256
TEMPERATURE = 0.0

# --- Sensibilité dataset ildpil ---
SENSITIVE_LABELS       = ("HEALTH", "POLITICS", "ETHNIC", "SEX", "BELIEF")
NOT_SENSITIVE_LABEL    = "NOT_CONFIDENTIAL"
SENSITIVE_ENTITY_TYPES = ("PERSON", "DEM", "MISC", "ORG", "LOC")

# --- Génération de queries ---
BATCH_SIZE   = 5     # documents par appel LLM
TEXT_EXCERPT = 600   # caractères du document envoyés au LLM
RANDOM_SEED  = 42
