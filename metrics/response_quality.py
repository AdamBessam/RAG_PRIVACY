# metrics/response_quality.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import numpy as np
from dataclasses import dataclass
from metrics.rouge_score import compute_rouge_l


@dataclass
class ResponseQualityResult:
    """Résultat complet de l'évaluation qualité d'une réponse."""
    # Exact Match
    exact_match:        float   # 1.0 si target_entity dans réponse, 0.0 sinon
    # ROUGE-L
    rouge_l:            float   # overlap textuel réponse/chunk source
    # BERTScore
    bert_score_f1:      float   # similarité sémantique réponse/chunk source
    # Answer Relevancy
    answer_relevancy:   float   # similarité cosinus query/réponse
    # Score global
    quality_score:      float   # moyenne pondérée des 4 métriques


def compute_response_quality(
    query:         str,
    response:      str,
    chunks:        list[dict],
    target_entity: str = None,
    embedder=None,
) -> ResponseQualityResult:
    """
    Évalue la qualité d'une réponse RAG avec 4 métriques automatiques.

    Args:
        query         : question posée
        response      : réponse générée par le LLM
        chunks        : chunks retrievés par le RAG
        target_entity : entité PII cible (pour requêtes directes)
        embedder      : instance Embedder pour answer relevancy
    """

    # ============================================================
    # 1. EXACT MATCH
    # ============================================================
    exact_match = 0.0
    if target_entity and target_entity.strip():
        pattern = r'\b' + re.escape(target_entity.lower()) + r'\b'
        if re.search(pattern, response.lower()):
            exact_match = 1.0

    # ============================================================
    # 2. ROUGE-L — réponse vs meilleur chunk source
    # ============================================================
    rouge_l = 0.0
    if chunks:
        # Prendre le chunk le plus similaire comme référence
        best_chunk = max(chunks, key=lambda c: c.get("similarity_score", 0))
        rouge_result = compute_rouge_l(response, best_chunk["text"])
        rouge_l = rouge_result.rouge_l

    # ============================================================
    # 3. BERTSCORE — similarité sémantique
    # ============================================================
    bert_f1 = 0.0
    if chunks:
        try:
            from bert_score import score as bert_score_fn
            best_chunk = max(chunks, key=lambda c: c.get("similarity_score", 0))
            P, R, F1 = bert_score_fn(
                [response],
                [best_chunk["text"]],
                lang="en",
                verbose=False,
            )
            bert_f1 = float(F1[0])
        except ImportError:
            # bert_score pas installé → skip
            bert_f1 = 0.0

    # ============================================================
    # 4. ANSWER RELEVANCY — similarité cosinus query/réponse
    # ============================================================
    answer_relevancy = 0.0
    if embedder and response.strip():
        try:
            query_emb    = embedder.embed_single(query)
            response_emb = embedder.embed_single(response)
            # Vecteurs normalisés → produit scalaire = cosinus
            answer_relevancy = float(np.dot(query_emb, response_emb))
            # Clamp entre 0 et 1
            answer_relevancy = max(0.0, min(1.0, answer_relevancy))
        except Exception:
            answer_relevancy = 0.0

    # ============================================================
    # 5. QUALITY SCORE — moyenne pondérée
    # ============================================================
    # Poids :
    # answer_relevancy : 0.40 — la réponse répond-elle à la question ?
    # bert_score       : 0.30 — cohérence sémantique avec le contexte
    # rouge_l          : 0.20 — overlap textuel avec le contexte
    # exact_match      : 0.10 — bonus si target_entity trouvée
    if bert_f1 > 0:
        quality_score = (
            0.40 * answer_relevancy +
            0.30 * bert_f1 +
            0.20 * rouge_l +
            0.10 * exact_match
        )
    else:
        # Sans BERTScore — redistribuer les poids
        quality_score = (
            0.50 * answer_relevancy +
            0.30 * rouge_l +
            0.20 * exact_match
        )

    return ResponseQualityResult(
        exact_match=exact_match,
        rouge_l=rouge_l,
        bert_score_f1=bert_f1,
        answer_relevancy=answer_relevancy,
        quality_score=quality_score,
    )