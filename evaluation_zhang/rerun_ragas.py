"""Recalcule uniquement CR, SS, AR via RAGAS depuis les fichiers en cache."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA = Path(__file__).parent.parent / "data" / "zhang_eval_test"

attacks   = json.loads((DATA / "attacks_test.json").read_text())
responses = json.loads((DATA / "responses_test.json").read_text())
contexts  = json.loads((DATA / "contexts_test.json").read_text())
refs      = json.loads((DATA / "reference_responses_test.json").read_text())

from metric_utility import compute_utility

u = compute_utility(attacks, responses, contexts, refs)
print(f"CR={u['CR']:.4f}  SS={u['SS']:.4f}  AR={u['AR']:.4f}")
