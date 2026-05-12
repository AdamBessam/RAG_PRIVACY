# embeddings/embedder.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
import numpy as np
from tqdm import tqdm


class Embedder:
    """
    Wrapper autour de all-MiniLM-L6-v2.
    Produit des vecteurs de dimension 384.
    Fonctionne sur CPU sans GPU.
    """

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model loaded.")

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode une liste de textes.
        Retourne un array numpy de shape (n, 384).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # nécessaire pour similarité cosinus
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Encode un seul texte.
        Retourne un vecteur de shape (384,).
        """
        return self.model.encode(
            [text],
            normalize_embeddings=True,
        )[0]


if __name__ == "__main__":
    embedder = Embedder()

    # Test sur quelques phrases
    test_texts = [
        "Henrik Hasslund lodged an application against Denmark.",
        "The applicant was represented by a lawyer in Copenhagen.",
        "Personal data must be protected under GDPR regulations.",
    ]

    print(f"\n🔍 Test embed_texts sur {len(test_texts)} phrases :")
    vectors = embedder.embed_texts(test_texts)
    print(f"   Shape output    : {vectors.shape}")
    print(f"   Dimension       : {vectors.shape[1]}")
    print(f"   Norme vecteur 0 : {np.linalg.norm(vectors[0]):.4f} (doit être ~1.0)")

    print(f"\n🔍 Test embed_single :")
    v = embedder.embed_single("Test single embedding.")
    print(f"   Shape  : {v.shape}")
    print(f"   Norme  : {np.linalg.norm(v):.4f} (doit être ~1.0)")

    # Test similarité cosinus entre deux phrases
    from numpy import dot
    sim = dot(vectors[0], vectors[1])
    print(f"\n🔍 Similarité cosinus phrase 0 vs phrase 1 : {sim:.4f}")
    print(f"   (entre 0 et 1, plus c'est proche de 1, plus les phrases sont similaires)")