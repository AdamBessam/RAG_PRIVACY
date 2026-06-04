"""
CPB v2 — SADDetectorNV avec taxonomie enrichie dynamiquement depuis le store.

Amélioration vs countermeasure/sad_detector.py :
- Accepte extra_taxonomy dict[str, list[str]] construit au démarrage depuis
  les chunks du store (phrases réelles anonymisées par Presidio).
- Fusionne avec SENSITIVE_TAXONOMY de base → centroïdes SBERT adaptatifs.
- Aucune configuration manuelle de domaine requise.
- Fonctionne que le dataset soit annoté (GT) ou non (Presidio fallback).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from countermeasure.sad_detector import SADDetector, SENSITIVE_TAXONOMY


class SADDetectorNV(SADDetector):
    """
    CPB v2 block 6 — SADDetector with domain-adaptive taxonomy.

    extra_taxonomy: sentences extracted and anonymized from the store at startup,
    grouped by SAD category (e.g. "FINANCIAL", "IDENTITY").
    Merged with the base SENSITIVE_TAXONOMY so that SBERT centroids cover both
    the base (SAGE/legal) and domain-specific sensitive attributes.

    If extra_taxonomy is None or empty, behaves identically to SADDetector.
    """

    def __init__(
        self,
        extra_taxonomy: dict[str, list[str]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Deep copy base taxonomy to avoid mutating the module-level constant
        self._taxonomy: dict[str, list[str]] = {
            k: list(v) for k, v in SENSITIVE_TAXONOMY.items()
        }
        if extra_taxonomy:
            for category, sentences in extra_taxonomy.items():
                if not sentences:
                    continue
                if category in self._taxonomy:
                    self._taxonomy[category].extend(sentences)
                else:
                    self._taxonomy[category] = list(sentences)

    def _get_centroids(self, embedder) -> dict[str, np.ndarray]:
        """Override to use the enriched taxonomy instead of the base constant."""
        if self._centroids is None:
            self._centroids = {}
            for category, sentences in self._taxonomy.items():
                embs = embedder.embed_texts(sentences)
                centroid = embs.mean(axis=0)
                norm = np.linalg.norm(centroid)
                self._centroids[category] = centroid / (norm + 1e-9)
        return self._centroids
