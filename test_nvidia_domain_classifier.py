"""
Test autonome — compare nvidia/domain-classifier vs Llama 3.1 8B (étape 0b
actuelle de countermeasure_v3) sur les corpus ChromaDB existants.

But : décider lequel des deux détecte le mieux le domaine, AVANT toute
intégration dans countermeasure_v4. Aucune modification de countermeasure_v3
ni création de countermeasure_v4 ici — le _step_0b de Llama est importé
tel quel, sans changement.

Usage:
    python test_nvidia_domain_classifier.py
"""
import importlib.util
import io
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows (console cp1252 ne gère pas ≈ / ✅ / ⚠️)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

# Imports paresseux (chromadb/torch jamais au niveau module — cf. mémoire
# "Harness d'éval", crash natif Windows sinon).

DATASETS = [
    {
        "name": "rag_benchmark (ATB — légal)",
        "chroma_dir": "data/chroma_db",
        "collection": "rag_benchmark",
        "expected": "legal",
    },
    {
        "name": "financial_benchmark",
        "chroma_dir": "benchmark_financial/chroma_db",
        "collection": "financial_benchmark",
        "expected": "finance",
    },
    {
        "name": "asq_phi_benchmark (clinique synthétique)",
        "chroma_dir": "benchmark_asq_phi/chroma_db",
        "collection": "asq_phi_benchmark",
        "expected": "health",
    },
    {
        "name": "chatdoctor_eval_corpus",
        "chroma_dir": "data/chroma_chatdoctor",
        "collection": "chatdoctor_eval_corpus",
        "expected": "health",
    },
]

N_SAMPLES = 30
MAX_CHARS = 2000  # le tokenizer DeBERTa tronque à 512 tokens de toute façon


def load_classifier():
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from huggingface_hub import PyTorchModelHubMixin

    class CustomModel(nn.Module, PyTorchModelHubMixin):
        def __init__(self, config):
            super().__init__()
            self.model = AutoModel.from_pretrained(config["base_model"], dtype=torch.float32)
            self.dropout = nn.Dropout(config["fc_dropout"])
            self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

        def forward(self, input_ids, attention_mask):
            features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            dropped = self.dropout(features)
            outputs = self.fc(dropped)
            return torch.softmax(outputs[:, 0, :], dim=1)

    print("Chargement de nvidia/domain-classifier (premier lancement = téléchargement HF)...")
    config = AutoConfig.from_pretrained("nvidia/domain-classifier")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/domain-classifier")
    model = CustomModel.from_pretrained("nvidia/domain-classifier")
    model.eval()
    print("Modèle chargé.\n")
    return torch, model, tokenizer, config


def load_llama_bootstrap():
    """Import direct de cpb_bootstrap_v3.py, en évitant countermeasure_v3/__init__.py
    (qui charge Presidio/GLiNER inutilement ici) — même technique que test_bootstrap.py.
    """
    repo_root = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "cpb_bootstrap_v3",
        str(repo_root / "countermeasure_v3" / "cpb_bootstrap_v3.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CPBBootstrapV3(store=None)


def sample_chunks(chroma_dir: str, collection_name: str, n: int) -> list[str]:
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(Path(chroma_dir)),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(collection_name)
    result = collection.get(limit=n, include=["documents"])
    docs = [d for d in (result.get("documents") or []) if d and d.strip()]
    print(f"  {len(docs)} chunks échantillonnés sur {collection.count()} dans '{collection_name}'")
    return docs


def classify(torch, model, tokenizer, config, texts: list[str]) -> list[tuple[str, float]]:
    truncated = [t[:MAX_CHARS] for t in texts]
    inputs = tokenizer(truncated, return_tensors="pt", padding="longest", truncation=True)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
    confidences, predicted = outputs.max(dim=1)
    return [
        (config.id2label[idx.item()], float(conf.item()))
        for idx, conf in zip(predicted, confidences)
    ]


def main():
    torch, model, tokenizer, config = load_classifier()
    llama_bootstrap = load_llama_bootstrap()

    summary = []
    for ds in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"  {ds['name']}  (attendu ≈ {ds['expected']})")
        print('=' * 70)

        chunks = sample_chunks(ds["chroma_dir"], ds["collection"], N_SAMPLES)
        if not chunks:
            print("  ⚠️  Aucun chunk trouvé, collection vide ou inaccessible — skip")
            summary.append((ds["name"], ds["expected"], "N/A", 0.0, 0.0, 0.0, "N/A", 0.0, 0.0))
            continue

        # ── nvidia/domain-classifier : classification par chunk + vote majoritaire ──
        t0 = time.perf_counter()
        predictions = classify(torch, model, tokenizer, config, chunks)
        nvidia_elapsed = time.perf_counter() - t0

        votes: dict[str, list[float]] = {}
        for label, conf in predictions:
            votes.setdefault(label, []).append(conf)

        print(f"\n  [nvidia/domain-classifier]  ({nvidia_elapsed:.1f}s pour {len(chunks)} chunks)")
        for label, confs in sorted(votes.items(), key=lambda kv: -len(kv[1])):
            mean_conf = sum(confs) / len(confs)
            print(f"    {label:30s}  {len(confs):3d}/{len(predictions)}  conf. moy={mean_conf:.2f}")

        nvidia_label = max(votes, key=lambda l: len(votes[l]))
        nvidia_vote_share = len(votes[nvidia_label]) / len(predictions)
        nvidia_conf = sum(votes[nvidia_label]) / len(votes[nvidia_label])
        print(f"    >>> Majorité : {nvidia_label}  ({nvidia_vote_share:.0%}, conf. moy={nvidia_conf:.2f})")

        # ── Llama 3.1 8B (countermeasure_v3._step_0b, inchangé) — 1 appel sur excerpt ──
        t0 = time.perf_counter()
        try:
            llama_domain, llama_conf = llama_bootstrap._step_0b(chunks)
        except Exception as exc:
            llama_domain, llama_conf = f"ERROR({exc})", 0.0
        llama_elapsed = time.perf_counter() - t0

        print(f"\n  [Llama 3.1 8B — _step_0b actuel]  ({llama_elapsed:.1f}s, 1 appel Ollama)")
        print(f"    >>> Domaine : {llama_domain}  (confiance déclarée {llama_conf:.2f})")

        summary.append((
            ds["name"], ds["expected"],
            nvidia_label, nvidia_vote_share, nvidia_conf, nvidia_elapsed,
            llama_domain, llama_conf, llama_elapsed,
        ))

    print(f"\n{'=' * 90}")
    print("  RÉSUMÉ COMPARATIF")
    print('=' * 90)
    for name, expected, nv_label, nv_share, nv_conf, nv_t, ll_domain, ll_conf, ll_t in summary:
        print(f"\n  {name}  (attendu ≈ {expected})")
        print(f"    nvidia/domain-classifier : {nv_label:25s}  vote={nv_share:.0%}  conf={nv_conf:.2f}  ({nv_t:.1f}s)")
        print(f"    Llama 3.1 8B             : {ll_domain:25s}  conf={ll_conf:.2f}  ({ll_t:.1f}s)")


if __name__ == "__main__":
    main()
