"""Debug autonome de la métrique PI sur le Naive RAG (1 instance)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metric_pi import PIMetric

DATA_DIR = Path(__file__).parent.parent / "data" / "zhang_eval"

responses = json.loads((DATA_DIR / "naive_responses.json").read_text())
attacks   = json.loads((DATA_DIR / "attack_queries.json").read_text())
assert len(responses) == len(attacks)

i = 0
response = responses[i]
doc_id = attacks[i]["doc_id"]

print(f"Instance {i} — doc_id attendu = {doc_id}")
print("Réponse (début) :", response[:200].replace("\n", " "), "...")
print("=" * 60)

PIMetric().debug_pi(response, doc_id)
