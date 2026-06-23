"""PI Naive agrégé sur les 300 instances, avec les poids actuels. Naive uniquement."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metric_pi import PIMetric

DATA_DIR = Path(__file__).parent.parent / "data" / "zhang_eval"
responses = json.loads((DATA_DIR / "naive_responses.json").read_text())
attacks = json.loads((DATA_DIR / "attack_queries.json").read_text())
assert len(responses) == len(attacks)

m = PIMetric()
scores = []
for i, (resp, atk) in enumerate(zip(responses, attacks)):
    s = m.compute_pi(resp, atk["doc_id"])
    scores.append(s)
    print(f"  [{i+1}/{len(responses)}] PI={s:.3f}      ", end="\r")
print()

mean = sum(scores) / len(scores)
hits = sum(1 for s in scores if s > 0)
print(f"Naive PI (moyenne sur {len(scores)}) : {mean:.4f}")
print(f"Instances avec PI > 0 : {hits}/{len(scores)}")
