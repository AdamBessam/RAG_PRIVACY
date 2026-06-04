"""
Utilitaires partagés pour le benchmark ASQ-PHI.
"""
import json


def parse_asq_phi(raw_text: str) -> list[dict]:
    """
    Parse le format ASQ-PHI :
        ===QUERY===
        <texte de la query clinique>
        ===PHI_TAGS===
        {"identifier_type": "...", "value": "..."}
        ...

    Retourne une liste de dicts avec query, phi_entities, has_phi.
    """
    entries = []
    blocks  = raw_text.strip().split("===QUERY===")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        parts = block.split("===PHI_TAGS===")
        if len(parts) != 2:
            continue

        query_text   = parts[0].strip()
        phi_tags_raw = parts[1].strip()

        if not query_text:
            continue

        phi_entities = []
        for line in phi_tags_raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                tag   = json.loads(line)
                value = tag.get("value", "").strip()
                if value:
                    start = query_text.lower().find(value.lower())
                    phi_entities.append({
                        "entity_type": tag.get("identifier_type", "UNKNOWN").upper(),
                        "text":        value,
                        "start":       start,
                        "end":         start + len(value) if start != -1 else -1,
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        entries.append({
            "query":        query_text,
            "phi_entities": phi_entities,
            "has_phi":      len(phi_entities) > 0,
        })

    return entries
