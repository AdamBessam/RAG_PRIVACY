# data/chunker.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
from tqdm import tqdm
from config import CHUNK_SIZE, CHUNK_OVERLAP, SPACY_MODEL


def load_spacy_model():
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        print(f"⚠️  Modèle spaCy '{SPACY_MODEL}' non trouvé. Téléchargement...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL])
        return spacy.load(SPACY_MODEL)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Découpe chaque document en chunks.
    Garantit qu'aucune entité PII n'est coupée entre deux chunks.
    """
    nlp = load_spacy_model()
    nlp.max_length = 2_000_000

    all_chunks = []
    chunk_counter = 0

    for doc in tqdm(documents, desc="Chunking documents"):
        text = doc["text"]
        chunks = _split_text(text, nlp, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_text, char_start, char_end in chunks:
            chunk_pii = _filter_pii(doc["pii_entities"], char_start, char_end)

            all_chunks.append({
                "chunk_id":     f"{doc['doc_id']}_c{chunk_counter}",
                "doc_id":       doc["doc_id"],
                "text":         chunk_text,
                "char_start":   char_start,
                "char_end":     char_end,
                "pii_entities": chunk_pii,
            })
            chunk_counter += 1

    print(f"✅ {len(all_chunks)} chunks produits depuis {len(documents)} documents")
    print(f"   Moyenne : {len(all_chunks) / len(documents):.1f} chunks/document")
    return all_chunks


def _split_text(text: str, nlp, chunk_size: int, overlap: int) -> list[tuple]:
    """
    Découpe un texte en chunks de ~chunk_size caractères,
    en coupant uniquement aux frontières de phrases (spaCy).
    Retourne une liste de (chunk_text, char_start, char_end).
    """
    doc = nlp(text, disable=["ner", "lemmatizer"])
    sentences = list(doc.sents)

    chunks = []
    current_sentences = []
    current_length = 0

    for sent in sentences:
        sent_len = len(sent.text)

        if current_length + sent_len > chunk_size and current_sentences:
            # Ferme le chunk courant
            char_start = current_sentences[0].start_char
            char_end   = current_sentences[-1].end_char
            chunk_text = text[char_start:char_end]  # texte original exact, pas de join
            chunks.append((chunk_text, char_start, char_end))

            # Overlap
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s.text) <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s.text)
                else:
                    break

            current_sentences = overlap_sents
            current_length = overlap_len

        current_sentences.append(sent)
        current_length += sent_len

    # Dernier chunk
    if current_sentences:
        char_start = current_sentences[0].start_char
        char_end   = current_sentences[-1].end_char
        chunk_text = text[char_start:char_end]  # texte original exact
        chunks.append((chunk_text, char_start, char_end))

    return chunks


def _filter_pii(pii_entities: list[dict], start: int, end: int) -> list[dict]:
    """
    Retourne uniquement les entités PII dont les offsets sont
    entièrement contenus dans [start, end].
    """
    return [
        ent for ent in pii_entities
        if ent["start"] >= start and ent["end"] <= end
    ]


if __name__ == "__main__":
    from data.loader import load_raw_documents

    documents = load_raw_documents()
    chunks = chunk_documents(documents)

    # Vérification manuelle
    print(f"\n🔍 Exemple — premier chunk :")
    c = chunks[0]
    print(f"   chunk_id   : {c['chunk_id']}")
    print(f"   doc_id     : {c['doc_id']}")
    print(f"   char_start : {c['char_start']}")
    print(f"   char_end   : {c['char_end']}")
    print(f"   nb PII     : {len(c['pii_entities'])}")
    print(f"   texte      : {c['text'][:120]}...")

    # Vérification anti-coupure PII
    print(f"\n✅ Vérification anti-coupure PII :")
    violations = 0
    for chunk in chunks:
        for ent in chunk["pii_entities"]:
            if ent["text"].strip() not in chunk["text"]:
                violations += 1
                print(f"   ⚠️  PII coupée :")
                print(f"       PII text   : '{ent['text']}'")
                print(f"       PII type   : {ent['type']}")
                print(f"       chunk_id   : {chunk['chunk_id']}")
                print(f"       chunk text : '{chunk['text'][:100]}...'")
    if violations == 0:
        print(f"   Aucune PII coupée sur {len(chunks)} chunks ✓")
    else:
        print(f"   ⚠️  {violations} PII coupées détectées !")