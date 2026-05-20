import sys
import json
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_text_from_path(file_path: str) -> str:
    """Charge un fichier depuis un chemin local."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    return _parse(path.suffix.lower(), path.read_bytes())


def load_text_from_upload(content: bytes, filename: str) -> str:
    """Charge un fichier depuis un upload Streamlit (bytes)."""
    suffix = Path(filename).suffix.lower()
    return _parse(suffix, content)


def _parse(suffix: str, content: bytes) -> str:
    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore")

    if suffix == ".json":
        data = json.loads(content.decode("utf-8"))
        return _json_to_text(data)

    if suffix == ".csv":
        import pandas as pd
        import io
        df = pd.read_csv(io.BytesIO(content))
        return "\n".join(df.astype(str).apply(" ".join, axis=1))

    if suffix in (".xlsx", ".xls"):
        import pandas as pd
        import io
        df = pd.read_excel(io.BytesIO(content))
        return "\n".join(df.astype(str).apply(" ".join, axis=1))

    if suffix == ".pdf":
        import io
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf n'est pas installé. Exécutez : pip install pypdf"
            )
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    # Fallback : tente de décoder comme texte brut
    return content.decode("utf-8", errors="ignore")


def _json_to_text(data) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(" ".join(str(v) for v in item.values() if v))
        return "\n\n".join(parts)
    if isinstance(data, dict):
        return " ".join(str(v) for v in data.values() if v)
    return str(data)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Découpe un texte en chunks avec chevauchement."""
    doc_id = uuid.uuid4().hex[:8]
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append({
                "chunk_id": f"{doc_id}_c{idx}",
                "doc_id": doc_id,
                "text": piece,
                "char_start": start,
                "char_end": end,
                "pii_entities": [],
            })
            idx += 1
        start += chunk_size - overlap

    return chunks
