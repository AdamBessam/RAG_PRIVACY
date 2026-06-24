# evaluation_zhang_openai/openai_embedder.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_PRICE_PER_TOKEN


class OpenAIEmbedder:
    """
    Wrapper autour de text-embedding-3-small (API OpenAI).
    Produit des vecteurs de dimension 1536.
    Même interface que embeddings.embedder.Embedder (remplacement direct).
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_EMBEDDING_MODEL
        self.total_tokens = 0

    @property
    def total_cost_usd(self) -> float:
        return self.total_tokens * OPENAI_EMBEDDING_PRICE_PER_TOKEN

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """
        Encode une liste de textes par batchs.
        Retourne un array numpy de shape (n, 1536).
        """
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend(d.embedding for d in response.data)
            self.total_tokens += response.usage.total_tokens
        return np.array(all_embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Encode un seul texte.
        Retourne un vecteur de shape (1536,).
        """
        response = self.client.embeddings.create(model=self.model, input=[text])
        self.total_tokens += response.usage.total_tokens
        return np.array(response.data[0].embedding)
