"""
Lanceur des interfaces Streamlit — contourne le crash Windows torch / pyarrow.

Problème (Windows) : `streamlit run app.py` charge pyarrow (via le bootstrap de
Streamlit) AVANT d'exécuter le script utilisateur. Or, si pyarrow est initialisé
avant torch, l'init de c10.dll échoue → OSError [WinError 1114]. Impossible donc
de forcer l'ordre « torch d'abord » depuis l'intérieur du script.

Solution : ce lanceur importe torch en TOUT PREMIER dans le process principal,
puis démarre Streamlit dans ce même process. torch initialise c10.dll avant que
pyarrow ne soit chargé → plus de crash. (Ordre validé : torch → streamlit → pyarrow.)

Usage (avec le Python du venv) :
    venv\\Scripts\\python.exe interface\\run.py              # interface 1 : ingestion (défaut)
    venv\\Scripts\\python.exe interface\\run.py chat          # interface 2 : chat CPB v6
    venv\\Scripts\\python.exe interface\\run.py <chemin.py>   # n'importe quelle app Streamlit
    # ou simplement, si le venv est activé :
    python interface/run.py chat
"""

import sys
from pathlib import Path

# 1) torch EN PREMIER : initialise c10.dll avant que Streamlit ne charge pyarrow.
try:
    import torch  # noqa: F401
except ImportError:
    pass

# 2) Résout l'app à lancer (alias court, ou chemin explicite).
ROOT = Path(__file__).parent.parent
APPS = {
    "ingestion": ROOT / "interface" / "app_ingestion_b0.py",
    "b0": ROOT / "interface" / "app_ingestion_b0.py",
    "chat": ROOT / "interface" / "app_chat_v5.py",
    "v5": ROOT / "interface" / "app_chat_v5.py",
    "v6": ROOT / "interface" / "app_chat_v5.py",
}


def _resolve_app(arg: str | None) -> Path:
    if not arg:
        return APPS["ingestion"]
    if arg in APPS:
        return APPS[arg]
    return Path(arg)  # chemin explicite vers une app Streamlit


if __name__ == "__main__":
    app = _resolve_app(sys.argv[1] if len(sys.argv) > 1 else None)
    if not app.exists():
        sys.exit(f"App introuvable : {app}")

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(app)]
    sys.exit(stcli.main())
