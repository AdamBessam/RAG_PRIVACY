"""
run_metrics_post_gen_masking.py — Variante ISOLÉE de run_metrics_by_query_type_cedh.py
où l'anonymisation des données sensibles se fait APRÈS la génération par le LLM.

Différence clé avec le script d'origine :
  • Pipeline d'origine (combo) : retrieve → MASQUER les chunks (B1/B3/B4) → LLM → B6.
    Le LLM ne voit JAMAIS les données brutes.
  • Ici (post-gen)             : retrieve → LLM voit les chunks BRUTS (meilleure réponse)
    → on masque ensuite la RÉPONSE avec Presidio (analyzer + anonymizer).
    On teste ainsi le PLACEMENT du masquage (après plutôt qu'avant).
  NB : Presidio masque noms/dates/lieux/identifiants (→ [PERSON_1]...), PAS le contenu
  libre (ex. un diagnostic) → attends-toi à une utilité haute mais une fuite d'attribut
  possible. C'est justement ce que les métriques ci-dessous mesurent.

En plus des 5 métriques (PII / QS / AR / RL / EM), ce script enregistre PAR REQUÊTE :
  • le TEMPS de réponse : retrieve_s, generate_s, mask_s, total_s.
  • la CONSOMMATION de RESSOURCES : RSS mémoire (Mo) avant/après + pic, CPU (s) via psutil.

Isolation / rollback : fichier autonome dans countermeasure_v5/. Il n'override AUCUNE
classe (il appelle cpb.naive_rag.retrieve/generate + cpb.pii_analyzer/pii_anonymizer
directement). Sorties dans un dossier DÉDIÉ `data/{dataset}_metrics_post_gen_masking/`
→ n'écrase aucun run existant. Pour revenir en arrière : supprimer ce fichier.

100 % LOCAL : génération Llama locale + métriques locales → AUCUN token OpenAI.
Un SEUL bootstrap B0 pour tout le run.

Usage (depuis la racine du repo) :
  python countermeasure_v5/run_metrics_post_gen_masking.py --per-type 100
  python countermeasure_v5/run_metrics_post_gen_masking.py --dataset financial --per-type 100
  python countermeasure_v5/run_metrics_post_gen_masking.py --per-type 50 --compare
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path
from time import perf_counter

try:
    import psutil
    _PROC = psutil.Process()
except Exception:                      # psutil absent → ressources = null (le run continue)
    psutil = None
    _PROC = None

# Sonde GPU globale (initialisée dans main selon --gpu-index / --no-gpu / --gpu-interval).
_GPU: "_GpuBackend | None" = None
_GPU_INTERVAL: float = 0.15

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent

ENTITY_HINT_RE = re.compile(r"^(.*) \([A-Z_]+\)$")
METRIC_KEYS = ("PII", "QS", "AR", "RL", "EM")

# Registre des datasets : config + store + métrique PII adaptée + dossier de sortie.
# out_subdir DÉDIÉ post-gen → on n'écrase jamais les runs du script d'origine.
DATASETS = {
    "cedh": {
        "config_module": "test_contre_mesure_ildpiltest.config",
        "store_module":  "test_contre_mesure_ildpiltest._store",
        "store_class":   "IldpilTestStore",
        "out_subdir":    "cedh_metrics_post_gen_masking",
        "pii_mode":      "cedh",        # PII sensibles (sensitivity ildpil)
    },
    "financial": {
        "config_module": "benchmark_financial.config",
        "store_module":  "benchmark_financial._store",
        "store_class":   "FinancialStore",
        "out_subdir":    "financial_metrics_post_gen_masking",
        "pii_mode":      "groundtruth", # toutes les PII annotées (pas de sensitivity)
    },
}


# ── Mesure des ressources (best-effort, jamais bloquant) ─────────────────────
def _rss_mb() -> float | None:
    """Mémoire résidente du process (Mo). None si psutil absent."""
    if _PROC is None:
        return None
    try:
        return _PROC.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _cpu_s() -> float | None:
    """Temps CPU cumulé (user+system, en s) du process. None si psutil absent."""
    if _PROC is None:
        return None
    try:
        t = _PROC.cpu_times()
        return float(t.user + t.system)
    except Exception:
        return None


# ── Sonde GPU (NVIDIA) : pynvml si dispo, sinon nvidia-smi, sinon inactive ────
# IMPORTANT : le LLM (Ollama) tourne dans un PROCESS SÉPARÉ sur le GPU → psutil ne
# le voit pas. On mesure donc l'util. GPU et la VRAM AU NIVEAU DE LA CARTE. Sur un
# GPU PARTAGÉ, ces valeurs incluent la charge des autres process (à interpréter).
class _GpuBackend:
    def __init__(self, index: int = 0):
        self.index = index
        self.mode: str | None = None
        self.handle = None
        self._pynvml = None
        self.total_mb: float | None = None
        self.name: str | None = None
        # 1) pynvml (rapide, in-process)
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._pynvml = pynvml
            self.mode = "pynvml"
            self.total_mb = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total / (1024 * 1024)
            raw = pynvml.nvmlDeviceGetName(self.handle)
            self.name = raw.decode() if isinstance(raw, bytes) else str(raw)
            return
        except Exception:
            self._pynvml = None
        # 2) nvidia-smi (toujours présent sur une machine GPU NVIDIA)
        if shutil.which("nvidia-smi"):
            self.mode = "smi"
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={index}",
                     "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                name, total = out.split(",")
                self.name, self.total_mb = name.strip(), float(total)
            except Exception:
                pass

    def available(self) -> bool:
        return self.mode is not None

    def sample(self) -> tuple[float | None, float | None]:
        """(util_gpu_%, vram_utilisée_Mo) ou (None, None) en cas d'échec."""
        if self.mode == "pynvml":
            try:
                p = self._pynvml
                u = p.nvmlDeviceGetUtilizationRates(self.handle).gpu
                m = p.nvmlDeviceGetMemoryInfo(self.handle).used / (1024 * 1024)
                return float(u), float(m)
            except Exception:
                return None, None
        if self.mode == "smi":
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.index}",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                u, m = out.split(",")
                return float(u), float(m)
            except Exception:
                return None, None
        return None, None


class _GpuSampler:
    """Context manager : échantillonne le GPU en tâche de fond pendant l'appel.
    Sur `with`, un thread lit util+VRAM toutes les `interval` s jusqu'à la sortie."""
    def __init__(self, backend: "_GpuBackend | None", interval: float | None = None):
        self.backend = backend
        self.interval = interval if interval is not None else _GPU_INTERVAL
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.utils: list[float] = []
        self.vrams: list[float] = []

    def __enter__(self):
        if self.backend and self.backend.available():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            u, m = self.backend.sample()
            if u is not None:
                self.utils.append(u)
            if m is not None:
                self.vrams.append(m)
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return False

    def result(self) -> dict:
        return {
            "gpu_util_pct_mean": _mean(self.utils),
            "gpu_util_pct_peak": max(self.utils) if self.utils else None,
            "gpu_vram_mb_peak":  max(self.vrams) if self.vrams else None,
            "gpu_vram_mb_mean":  _mean(self.vrams),
            "gpu_samples":       len(self.utils),
        }


def load_dataset(name: str) -> dict:
    """Charge dynamiquement config + store + paramètres du dataset choisi."""
    import importlib
    if name not in DATASETS:
        raise SystemExit(f"Dataset inconnu '{name}'. Choix : {list(DATASETS)}")
    spec = DATASETS[name]
    cfg = importlib.import_module(spec["config_module"])
    store_cls = getattr(importlib.import_module(spec["store_module"]), spec["store_class"])
    out_dir = ROOT / "data" / spec["out_subdir"]
    return {
        "name":         name,
        "chroma_dir":   cfg.CHROMA_DIR,
        "collection":   cfg.COLLECTION_NAME,
        "queries_file": cfg.QUERIES_FILE,
        "top_k":        getattr(cfg, "TOP_K", 5),
        "store_cls":    store_cls,
        "pii_mode":     spec["pii_mode"],
        "out_dir":      out_dir,
        "paths": {
            "results":   out_dir / "results.json",
            "responses": out_dir / "responses.json",
            "bootstrap": out_dir / "bootstrap_b0.json",
            "compare":   out_dir / "compare_postgen_vs_combo.json",
        },
    }


def parse_target_entity(q: dict) -> str | None:
    # 1) hint structuré "texte (TYPE)" (CEDH)
    for key in ("target_entity", "entity_hint", "entity", "target"):
        hint = q.get(key)
        if isinstance(hint, str):
            m = ENTITY_HINT_RE.match(hint)
            if m:
                return m.group(1).strip()
    # 2) champ 'company' en clair (financier) → cible de l'EM
    comp = q.get("company")
    if isinstance(comp, str) and comp.strip():
        return comp.strip()
    return None


def pii_leakage_rate(response: str, chunks: list[dict], query: str, mode: str) -> float:
    """Taux de fuite PII, adapté au dataset (les DEUX sont annotés) :
      - mode 'cedh'        : entités SENSIBLES (sensitivity ildpil), via compute_pii_leakage.
      - mode 'groundtruth' : TOUTES les PII annotées des chunks (financier), matching exact
                             sur le texte, en excluant les PII déjà présentes dans la question."""
    if mode == "cedh":
        from metrics.pii_leakage import compute_pii_leakage
        return compute_pii_leakage(response, chunks, query).leakage_rate

    # Ground-truth : toutes les PII annotées, exclusion de celles déjà dans la question.
    q = (query or "").lower()
    resp = (response or "").lower()
    texts: set[str] = set()
    for ch in chunks or []:
        pii = ch.get("pii_entities", [])
        if isinstance(pii, str):
            try:
                pii = json.loads(pii)
            except (json.JSONDecodeError, TypeError):
                pii = []
        for ent in pii:
            t = (ent.get("text") or "").strip().lower()
            if len(t) > 2 and t not in q:      # exclut les PII déjà connues de l'utilisateur
                texts.add(t)
    if not texts:
        return 0.0
    leaked = sum(1 for t in texts if t in resp)
    return leaked / len(texts)


def get_query_text(q: dict) -> str:
    text = q.get("query", "")
    if isinstance(text, dict):
        return text.get("query") or str(text)
    return text if isinstance(text, str) else str(text)


def load_queries_by_type(queries_file, per_type: int, seed: int) -> dict[str, list[dict]]:
    """Échantillonne `per_type` requêtes PAR query_type depuis le corpus 1000."""
    with open(queries_file, encoding="utf-8") as f:
        all_queries = json.load(f)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in all_queries:
        by_type[q.get("query_type", "unknown")].append(q)
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for qtype in sorted(by_type):
        items = by_type[qtype][:]
        rng.shuffle(items)
        out[qtype] = items[:per_type]
    return out


def build_cpb(ds: dict, mask_min_weight: float, use_domain_hints: bool,
              use_llm_combos: bool, dedup: bool):
    """Construit UNE fois la contre-mesure (un seul bootstrap B0).
    Retrieval = HybridRAG (dense ChromaDB + BM25, fusion RRF). Store selon le dataset.
    En mode post-gen on n'utilise PAS cpb.run() (on appelle retrieve/generate + Presidio
    à la main), mais on réutilise cpb.pii_analyzer / cpb.pii_anonymizer / cpb.naive_rag."""
    from countermeasure_v5.cpb_naive_rag_v5_combo import CPBNaiveRAGV5Combo
    from llms.llama_llm import LlamaLLM
    from rag.hybrid_rag import HybridRAG

    store = ds["store_cls"](chroma_dir=ds["chroma_dir"], collection_name=ds["collection"])
    llm = LlamaLLM()
    # HybridRAG : dense + BM25 + RRF. dedup=False → plusieurs chunks/doc (config hybrid_nodedup).
    hybrid = HybridRAG(store=store, llm=llm, dedup=dedup)
    return CPBNaiveRAGV5Combo(
        naive_rag=hybrid,
        mask_min_weight=mask_min_weight,
        use_domain_hints=use_domain_hints,
        use_llm_combos=use_llm_combos,
    )


def _chunk_texts(chunks: list) -> list[str]:
    """Texte de chaque chunk (masqué si dict CPB, sinon str brut)."""
    out = []
    for c in chunks or []:
        if isinstance(c, dict):
            out.append(c.get("text", ""))
        else:
            out.append(str(c))
    return out


def _sanitize_chunks(chunks: list) -> list[dict]:
    """HybridRAG : les hits BM25 ont similarity_score=None → max() plante dans
    compute_response_quality. On remplace None par le rrf_score (ou 0.0), sur une
    COPIE, sans muter les chunks d'origine ni toucher au code métrique partagé."""
    safe = []
    for c in chunks or []:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        if cc.get("similarity_score") is None:
            cc["similarity_score"] = float(cc.get("rrf_score") or 0.0)
        safe.append(cc)
    return safe


# ── Les deux variantes de génération (renvoient un dict standardisé + timing) ─
def run_post_gen(cpb, qtext: str, top_k: int) -> dict:
    """MASQUAGE APRÈS GÉNÉRATION :
      1. retrieve → chunks BRUTS ;
      2. le LLM génère à partir des chunks BRUTS (aucun masquage du contexte) ;
      3. Presidio (analyzer + anonymizer) masque la RÉPONSE.
    B1/B2/B4-sur-chunks/B6 sont contournés (test pur du placement du masquage)."""
    rag = cpb.naive_rag
    t0 = perf_counter()
    raw_chunks = rag.retrieve(qtext, top_k=top_k)
    t1 = perf_counter()
    llm_response = rag.generate(qtext, raw_chunks)
    raw_text = llm_response.response
    t2 = perf_counter()
    pii = cpb.pii_analyzer.analyze(raw_text)
    if pii.findings:
        masked_text, n_repl = cpb.pii_anonymizer.anonymize_text(raw_text, pii.findings)
    else:
        masked_text, n_repl = raw_text, 0
    t3 = perf_counter()
    return {
        "response":                 masked_text,
        "response_before_masking":  raw_text,      # ce que le LLM a vraiment produit
        "raw_chunks":               raw_chunks,
        "masked_context":           _chunk_texts(raw_chunks),  # LLM a vu les chunks BRUTS
        "masked_query":             qtext,         # pas de masquage de la requête en post-gen
        "n_replacements":           n_repl,
        "timing": {
            "retrieve_s": t1 - t0,
            "generate_s": t2 - t1,
            "mask_s":     t3 - t2,
            "total_s":    t3 - t0,
        },
    }


def run_combo(cpb, qtext: str, top_k: int) -> dict:
    """MASQUAGE AVANT GÉNÉRATION (pipeline combo d'origine) : cpb.run() masque les
    chunks puis génère. Utilisé seulement en --compare (référence pre-gen)."""
    t0 = perf_counter()
    result = cpb.run(qtext, top_k=top_k)
    t1 = perf_counter()
    return {
        "response":                 result["response"],
        "response_before_masking":  None,
        "raw_chunks":               result.get("raw_chunks", []),
        "masked_context":           _chunk_texts(result.get("chunks", [])),
        "masked_query":             result.get("cpb_masked_query", qtext),
        "n_replacements":           None,
        "timing": {
            "retrieve_s": None,
            "generate_s": None,
            "mask_s":     None,
            "total_s":    t1 - t0,
        },
    }


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def score_group(runner, cpb, qtype: str, queries: list[dict], embedder, ds: dict) -> tuple[dict, list[dict]]:
    """Génère + score un groupe de requêtes (un query_type) avec la variante `runner`.
    Renvoie (métriques + perf agrégées, liste des enregistrements par requête)."""
    from metrics.response_quality import compute_response_quality

    top_k = ds["top_k"]
    pii_mode = ds["pii_mode"]
    agg = {k: 0.0 for k in METRIC_KEYS}
    totals, gens, masks, cpus = [], [], [], []
    gpu_util_means, gpu_util_peaks, gpu_vram_peaks = [], [], []
    peak_rss = 0.0
    records: list[dict] = []
    n = 0
    for i, q in enumerate(queries):
        qtext = get_query_text(q)
        print(f"      [{i + 1}/{len(queries)}] {q.get('global_id', q.get('query_id', ''))}...", end="\r")

        rss_before, cpu_before = _rss_mb(), _cpu_s()
        # Échantillonnage GPU en tâche de fond PENDANT la requête (couvre la génération).
        with _GpuSampler(_GPU) as gpu:
            try:
                out = runner(cpb, qtext, top_k)
            except Exception as exc:
                out = {
                    "response": f"ERROR: {exc}", "response_before_masking": None,
                    "raw_chunks": [], "masked_context": [], "masked_query": qtext,
                    "n_replacements": None,
                    "timing": {"retrieve_s": None, "generate_s": None, "mask_s": None, "total_s": None},
                }
        gpu_res = gpu.result()
        rss_after, cpu_after = _rss_mb(), _cpu_s()

        cpu_s = (cpu_after - cpu_before) if (cpu_after is not None and cpu_before is not None) else None
        for v in (rss_before, rss_after):
            if v is not None:
                peak_rss = max(peak_rss, v)
        if gpu_res["gpu_util_pct_mean"] is not None:
            gpu_util_means.append(gpu_res["gpu_util_pct_mean"])
        if gpu_res["gpu_util_pct_peak"] is not None:
            gpu_util_peaks.append(gpu_res["gpu_util_pct_peak"])
        if gpu_res["gpu_vram_mb_peak"] is not None:
            gpu_vram_peaks.append(gpu_res["gpu_vram_mb_peak"])

        safe_chunks = _sanitize_chunks(out["raw_chunks"])
        response = out["response"]
        pii_rate = pii_leakage_rate(response, safe_chunks, qtext, pii_mode)
        rq = compute_response_quality(
            query=qtext, response=response, chunks=safe_chunks,
            target_entity=parse_target_entity(q), embedder=embedder,
            precomputed_bert_f1=0.0,   # BF1 désactivé → QS sur AR/RL/EM
        )
        metrics = {
            "PII": pii_rate, "QS": rq.quality_score,
            "AR": rq.answer_relevancy, "RL": rq.rouge_l, "EM": rq.exact_match,
        }
        for k in METRIC_KEYS:
            agg[k] += metrics[k]

        t = out["timing"]
        totals.append(t["total_s"]); gens.append(t["generate_s"]); masks.append(t["mask_s"])
        cpus.append(cpu_s)

        records.append({
            "global_id":               q.get("global_id", q.get("query_id", "")),
            "query_type":              qtype,
            "query":                   qtext,
            "target_entity":           parse_target_entity(q),
            "masked_query":            out["masked_query"],
            "response":                response,
            "response_before_masking": out["response_before_masking"],
            "n_replacements":          out["n_replacements"],
            "masked_context":          out["masked_context"],  # ce que le LLM a réellement vu
            "metrics":                 metrics,
            "timing":                  t,
            "resources": {
                "rss_mb_before": rss_before, "rss_mb_after": rss_after, "cpu_s": cpu_s,
                "gpu_util_pct_mean": gpu_res["gpu_util_pct_mean"],
                "gpu_util_pct_peak": gpu_res["gpu_util_pct_peak"],
                "gpu_vram_mb_peak":  gpu_res["gpu_vram_mb_peak"],
                "gpu_samples":       gpu_res["gpu_samples"],
            },
        })
        n += 1
    print()
    agg_out = {"n": n, **{k: (agg[k] / n if n else 0.0) for k in METRIC_KEYS}}
    agg_out["perf"] = {
        "total_s_mean":      _mean(totals),
        "generate_s_mean":   _mean(gens),
        "mask_s_mean":       _mean(masks),
        "cpu_s_mean":        _mean(cpus),
        "rss_mb_peak":       peak_rss or None,
        "gpu_util_pct_mean": _mean(gpu_util_means),
        "gpu_util_pct_peak": max(gpu_util_peaks) if gpu_util_peaks else None,
        "gpu_vram_mb_peak":  max(gpu_vram_peaks) if gpu_vram_peaks else None,
    }
    return agg_out, records


def score_all(runner, cpb, groups, embedder, ds) -> tuple[dict, list]:
    """Score tous les query_type ; renvoie (rows par type, tous les records)."""
    rows, records = {}, []
    for qtype, queries in groups.items():
        if not queries:
            continue
        print(f"  -> query_type = {qtype}  (n={len(queries)})")
        rows[qtype], recs = score_group(runner, cpb, qtype, queries, embedder, ds)
        records.extend(recs)
    return rows, records


def weighted_global(rows: dict) -> dict | None:
    tot_n = sum(r["n"] for r in rows.values())
    if not tot_n:
        return None
    g = {"n": tot_n, **{k: sum(r[k] * r["n"] for r in rows.values()) / tot_n for k in METRIC_KEYS}}
    # Perf globale : moyenne pondérée par n (temps, util. GPU moyenne), max (pics).
    mean_keys = ("total_s_mean", "generate_s_mean", "mask_s_mean", "cpu_s_mean", "gpu_util_pct_mean")
    max_keys = ("rss_mb_peak", "gpu_util_pct_peak", "gpu_vram_mb_peak")
    perf: dict = {}
    for pk in mean_keys:
        num = sum((r["perf"].get(pk) or 0.0) * r["n"] for r in rows.values()
                  if r.get("perf", {}).get(pk) is not None)
        den = sum(r["n"] for r in rows.values() if r.get("perf", {}).get(pk) is not None)
        perf[pk] = (num / den) if den else None
    for pk in max_keys:
        vals = [r["perf"].get(pk) for r in rows.values() if r.get("perf", {}).get(pk)]
        perf[pk] = max(vals) if vals else None
    g["perf"] = perf
    return g


def print_metrics_table(rows: dict, note: str = "") -> None:
    print("\n" + "=" * 74)
    print("  MÉTRIQUES PAR TYPE DE QUESTION  (PII ↓ = mieux, QS/AR/RL/EM ↑ = mieux)")
    if note:
        print(f"  {note}")
    print("=" * 74)
    print(f"  {'query_type':>11} {'n':>4} {'PII':>8} {'QS':>8} {'AR':>8} {'RL':>8} {'EM':>8}")
    print("-" * 74)
    for qtype, r in rows.items():
        print(f"  {qtype:>11} {r['n']:>4} {r['PII']:>8.4f} {r['QS']:>8.4f} "
              f"{r['AR']:>8.4f} {r['RL']:>8.4f} {r['EM']:>8.4f}")
    g = weighted_global(rows)
    if g:
        print("-" * 74)
        print(f"  {'GLOBAL':>11} {g['n']:>4} {g['PII']:>8.4f} {g['QS']:>8.4f} "
              f"{g['AR']:>8.4f} {g['RL']:>8.4f} {g['EM']:>8.4f}")
    print("=" * 74)


def print_perf_table(rows: dict, note: str = "") -> None:
    """Temps de réponse + ressources (CPU/RAM du process Python + GPU carte) par type.
    GPU util/VRAM = niveau CARTE (inclut Ollama, process séparé). '—' si GPU inactif."""
    W = 100
    print("\n" + "=" * W)
    print("  TEMPS DE RÉPONSE & RESSOURCES PAR TYPE  (moyennes ; temps s, mémoire Mo, GPU %)")
    if note:
        print(f"  {note}")
    print("=" * W)
    print(f"  {'query_type':>11} {'n':>4} {'total_s':>8} {'gen_s':>8} {'mask_s':>8} "
          f"{'cpu_s':>7} {'rss_pk':>8} {'gpu%_avg':>8} {'gpu%_pk':>8} {'vram_pk':>9}")
    print("-" * W)

    def fmt(v, w=8, p=3):
        return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"

    def row(label, n, p):
        print(f"  {label:>11} {n:>4} {fmt(p.get('total_s_mean'))} {fmt(p.get('generate_s_mean'))} "
              f"{fmt(p.get('mask_s_mean'))} {fmt(p.get('cpu_s_mean'), w=7)} {fmt(p.get('rss_mb_peak'), p=0)} "
              f"{fmt(p.get('gpu_util_pct_mean'), p=1)} {fmt(p.get('gpu_util_pct_peak'), p=1)} "
              f"{fmt(p.get('gpu_vram_mb_peak'), w=9, p=0)}")

    for qtype, r in rows.items():
        row(qtype, r["n"], r.get("perf", {}))
    g = weighted_global(rows)
    if g:
        print("-" * W)
        row("GLOBAL", g["n"], g["perf"])
    print("=" * W)


def print_banner(title: str) -> None:
    """Grande bannière pour séparer nettement chaque variante à l'écran."""
    line = "#" * 78
    print("\n" + line)
    print(f"#  {title}")
    print(line)


def print_compare_table(rows_post: dict, rows_combo: dict) -> None:
    """Comparaison POST-GEN (masquage après) vs COMBO (masquage avant). Δ = post − combo.
    ΔQS > 0 → masquer APRÈS préserve mieux l'utilité ;
    ΔPII > 0 → masquer APRÈS laisse (plus) fuiter (attribut non masqué par Presidio)."""
    print("\n" + "=" * 78)
    print("  POST-GEN (masquage après) vs COMBO (masquage avant)   Δ = post − combo")
    print("  ΔQS > 0 → post-gen garde + d'utilité | ΔPII > 0 → post-gen laisse + fuiter")
    print("=" * 78)
    print(f"  {'query_type':>11} | {'PII_post':>8} {'PII_comb':>8} {'ΔPII':>7} "
          f"| {'QS_post':>8} {'QS_comb':>8} {'ΔQS':>7}")
    print("-" * 78)
    for t in rows_post:
        a, b = rows_post[t], rows_combo.get(t, {})
        dpii = a["PII"] - b.get("PII", 0.0)
        dqs = a["QS"] - b.get("QS", 0.0)
        print(f"  {t:>11} | {a['PII']:>8.4f} {b.get('PII', 0):>8.4f} {dpii:>+7.4f} "
              f"| {a['QS']:>8.4f} {b.get('QS', 0):>8.4f} {dqs:>+7.4f}")
    ga, gb = weighted_global(rows_post), weighted_global(rows_combo)
    if ga and gb:
        print("-" * 78)
        print(f"  {'GLOBAL':>11} | {ga['PII']:>8.4f} {gb['PII']:>8.4f} {ga['PII'] - gb['PII']:>+7.4f} "
              f"| {ga['QS']:>8.4f} {gb['QS']:>8.4f} {ga['QS'] - gb['QS']:>+7.4f}")
    print("=" * 78)


def dump_bootstrap(cpb) -> dict:
    br = cpb.bootstrap_result
    return {
        "domain":             getattr(br, "domain", None),
        "domain_confidence":  getattr(br, "domain_confidence", None),
        "domain_source":      getattr(br, "domain_source", None),
        "used_fallback":      getattr(br, "used_fallback", None),
        "categories":         getattr(br, "dynamic_categories", []),
        "category_hints":     {k: sorted(v) for k, v in (getattr(br, "category_hints", {}) or {}).items()},
        "taxonomy":           {k: list(v) for k, v in (getattr(br, "dynamic_taxonomy", {}) or {}).items()},
        "learned_types":      sorted(getattr(br, "learned_types", set()) or set()),
        "risky_combinations": [sorted(c) for c in getattr(cpb, "risky_combos", [])],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Métriques CPB v5 avec MASQUAGE APRÈS GÉNÉRATION, ventilées par type "
                    "de question + temps de réponse & ressources (local, sans OpenAI).")
    parser.add_argument("--per-type", type=int, default=20,
                        help="Nb de requêtes échantillonnées par query_type (défaut 20).")
    parser.add_argument("--mask-min-weight", type=float, default=0.5,
                        help="Seuil v5 (n'affecte que la variante combo du --compare).")
    parser.add_argument("--no-domain-hints", action="store_true",
                        help="Désactive le Signal 2 (category_hints de B0) — combo du --compare.")
    parser.add_argument("--no-llm-combos", action="store_true",
                        help="Désactive la génération LLM des combinaisons (combo du --compare).")
    parser.add_argument("--dedup", action="store_true",
                        help="HybridRAG : 1 chunk par doc. Par défaut nodedup (plusieurs chunks/doc).")
    parser.add_argument("--compare", action="store_true",
                        help="Compare POST-GEN (masquage après) vs COMBO (masquage avant) sur les "
                             "MÊMES questions/B0 → effet du placement du masquage.")
    parser.add_argument("--dataset", default="cedh", choices=list(DATASETS),
                        help="Dataset à évaluer (défaut cedh). Détermine config/store/métrique PII.")
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="Index du GPU à échantillonner (défaut 0). Celui qu'utilise Ollama.")
    parser.add_argument("--gpu-interval", type=float, default=0.15,
                        help="Période d'échantillonnage GPU en s (défaut 0.15).")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Désactive la mesure GPU (util./VRAM).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Sonde GPU globale (LLM = Ollama sur GPU, process séparé → mesuré au niveau carte).
    global _GPU, _GPU_INTERVAL
    _GPU_INTERVAL = args.gpu_interval
    if not args.no_gpu:
        _GPU = _GpuBackend(index=args.gpu_index)

    ds = load_dataset(args.dataset)

    # En --compare on a besoin des combos générés (la variante COMBO/pre-gen).
    use_llm_combos = True if args.compare else (not args.no_llm_combos)

    print(f"=== CPB v5 — MASQUAGE APRÈS GÉNÉRATION — métriques PAR TYPE — {args.dataset.upper()} "
          f"(local, PII={ds['pii_mode']}) ===\n")
    if psutil is None:
        print("⚠  psutil indisponible → colonnes CPU/RAM à null. Installe psutil.\n")
    if args.no_gpu:
        print("ℹ  Mesure GPU désactivée (--no-gpu).\n")
    elif _GPU is not None and _GPU.available():
        print(f"ℹ  Sonde GPU active : backend={_GPU.mode}, gpu#{args.gpu_index}="
              f"{_GPU.name}, VRAM totale={_GPU.total_mb:.0f} Mo, "
              f"échantillonnage={_GPU_INTERVAL}s.\n")
    else:
        print("⚠  Aucun GPU détecté (ni pynvml ni nvidia-smi) → colonnes GPU à null.\n")
    groups = load_queries_by_type(ds["queries_file"], args.per_type, args.seed)
    total = sum(len(v) for v in groups.values())
    print(f"1. {total} requêtes : " + ", ".join(f"{t}={len(q)}" for t, q in groups.items()) + "\n")

    from embeddings.embedder import Embedder
    embedder = Embedder()

    print(f"2. Bootstrap CPB v5 combo + HybridRAG "
          f"({'dedup' if args.dedup else 'nodedup'})...")
    cpb = build_cpb(
        ds,
        mask_min_weight=args.mask_min_weight,
        use_domain_hints=not args.no_domain_hints,
        use_llm_combos=use_llm_combos,
        dedup=args.dedup,
    )

    # ── Décision B0 (commune aux deux variantes) ─────────────────────────────
    bootstrap_dump = dump_bootstrap(cpb)
    generated_combos = list(getattr(cpb, "risky_combos", []))
    retr = f"hybrid_{'dedup' if args.dedup else 'nodedup'}"

    print("\n── Décision B0 ──")
    print(f"  domaine={bootstrap_dump['domain']} "
          f"(conf={bootstrap_dump['domain_confidence']}, source={bootstrap_dump['domain_source']}, "
          f"fallback={bootstrap_dump['used_fallback']})")
    print(f"  catégories={bootstrap_dump['categories']}")
    print(f"  combinaisons risquées={bootstrap_dump['risky_combinations']}")

    out_dir = ds["out_dir"]
    paths = ds["paths"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # ── Comparaison : MÊMES questions/B0/retrieval, POST-GEN puis COMBO ────
        print_banner("VARIANTE 1/2 — POST-GEN (masquage APRÈS génération)")
        cpb.mask_all = False
        cpb.risky_combos = generated_combos
        rows_post, recs_post = score_all(run_post_gen, cpb, groups, embedder, ds)

        print_banner("VARIANTE 2/2 — COMBO (masquage AVANT génération, pipeline d'origine)")
        rows_combo, recs_combo = score_all(run_combo, cpb, groups, embedder, ds)

        print_banner("RÉSULTATS — VARIANTE 1/2 : POST-GEN")
        print_metrics_table(rows_post, note=f"POST-GEN — {args.dataset}, retrieval={retr}")
        print_perf_table(rows_post, note="POST-GEN — temps & ressources")

        print_banner("RÉSULTATS — VARIANTE 2/2 : COMBO")
        print_metrics_table(rows_combo, note=f"COMBO — {args.dataset}, retrieval={retr}")
        print_perf_table(rows_combo, note="COMBO — temps & ressources")

        print_banner("SYNTHÈSE — PLACEMENT DU MASQUAGE (POST-GEN vs COMBO)")
        print_compare_table(rows_post, rows_combo)

        with open(paths["compare"], "w", encoding="utf-8") as f:
            json.dump({
                "dataset": args.dataset,
                "pii_mode": ds["pii_mode"],
                "per_type": args.per_type,
                "retrieval": retr,
                "bootstrap_b0": bootstrap_dump,
                "post_gen": {"by_query_type": rows_post,  "global": weighted_global(rows_post)},
                "combo":    {"by_query_type": rows_combo, "global": weighted_global(rows_combo)},
            }, f, ensure_ascii=False, indent=2)
        with open(paths["responses"], "w", encoding="utf-8") as f:
            json.dump({
                "dataset": args.dataset,
                "per_type": args.per_type,
                "bootstrap_b0": bootstrap_dump,
                "responses_post_gen": recs_post,
                "responses_combo":    recs_combo,
            }, f, ensure_ascii=False, indent=2)
        with open(paths["bootstrap"], "w", encoding="utf-8") as f:
            json.dump(bootstrap_dump, f, ensure_ascii=False, indent=2)

        print("\nSauvegardé :")
        print(f"  comparaison post/combo → {paths['compare']}")
        print(f"  réponses (2 var.)      → {paths['responses']}")
        print(f"  décision B0            → {paths['bootstrap']}")
        return

    # ── Mode simple : POST-GEN seul ──────────────────────────────────────────
    print("\n3. Scoring par type de question — variante: POST-GEN (masquage après génération)\n")
    rows, all_records = score_all(run_post_gen, cpb, groups, embedder, ds)
    print_metrics_table(rows, note=f"{args.dataset}, POST-GEN, retrieval={retr}")
    print_perf_table(rows, note="POST-GEN — temps de réponse & ressources")

    with open(paths["results"], "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "pii_mode": ds["pii_mode"],
            "per_type": args.per_type,
            "variante": "post_gen_masking",
            "retrieval": retr,
            "bootstrap_b0": bootstrap_dump,
            "by_query_type": rows,
            "global": weighted_global(rows),
        }, f, ensure_ascii=False, indent=2)
    with open(paths["bootstrap"], "w", encoding="utf-8") as f:
        json.dump(bootstrap_dump, f, ensure_ascii=False, indent=2)
    with open(paths["responses"], "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "per_type": args.per_type,
            "variante": "post_gen_masking",
            "bootstrap_b0": bootstrap_dump,
            "responses": all_records,
        }, f, ensure_ascii=False, indent=2)

    print("\nSauvegardé :")
    print(f"  métriques par type → {paths['results']}")
    print(f"  décision B0        → {paths['bootstrap']}")
    print(f"  réponses générées  → {paths['responses']}  ({len(all_records)} réponses)")


if __name__ == "__main__":
    main()
