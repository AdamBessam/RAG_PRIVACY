"""
fix_cr_nan_chunks.py — One-time patch for chunk_*.json caches computed before
run_evaluation_chat_doctor.py mapped CR's NaN (0 retrieved contexts judged
relevant -> 0/0 in RAGAS's average-precision formula) to its real value, 0.0.

Rewrites each cached chunk in place. No RAGAS/LLM calls, so it costs nothing
and is safe to re-run.

Usage: python evaluation_chatdoctor/fix_cr_nan_chunks.py
"""

import json
import math
from pathlib import Path

UTILITY_CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chatdoctor_eval" / "utility_chunks"


def main():
    total_patched = 0
    for chunk_path in sorted(UTILITY_CHUNKS_DIR.glob("chunk_*.json")):
        rows = json.loads(chunk_path.read_text(encoding="utf-8"))
        patched = 0
        for row in rows:
            if math.isnan(row["CR"]):
                row["CR"] = 0.0
                patched += 1
        if patched:
            chunk_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            print(f"{chunk_path.name}: {patched} rows patched (CR NaN -> 0.0)")
        total_patched += patched
    print(f"\nTotal: {total_patched} rows patched")


if __name__ == "__main__":
    main()
