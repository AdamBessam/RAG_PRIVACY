# RAG Privacy — CPB v6

Contre-mesure **CPB (Contextual Privacy Budget) v6** contre la fuite de PII (données personnelles) dans les systèmes RAG (Retrieval-Augmented Generation), et son étude d'ablation sur corpus juridique (ECHR / CEDH).

Ce dépôt ne publie que la dernière itération de la contre-mesure (v6) et l'étude d'ablation qui la valide. Les versions précédentes (v1–v5) et les bancs de test annexes sont des étapes de recherche internes, non publiées ici.

## Contexte

Un système RAG répond aux questions d'un utilisateur en s'appuyant sur des documents récupérés dans une base vectorielle. Si ces documents contiennent des données personnelles (noms, données de santé, opinions politiques, etc.), un LLM peut les répéter dans sa réponse — que la question soit anodine, ciblée, ou une tentative de jailbreak. CPB est une contre-mesure qui s'insère autour d'un pipeline RAG pour limiter cette fuite sans détruire l'utilité des réponses.

## Architecture CPB v6

Rupture par rapport aux versions précédentes (v1–v5) : **la requête et les chunks récupérés ne sont jamais masqués**. Le LLM génère toujours à partir du contexte brut (meilleure utilité) ; seule la **réponse générée** est ensuite passée au crible.

```
requête utilisateur (brute)
        │
        ▼
  B1  QueryRiskScorer + BudgetGate — score de risque de la requête (NER, verbes
        │                            extractifs, regex jailbreak, risque multi-tour,
        │                            proximité sémantique) ; si risque trop élevé :
        │                            refus direct (pas de retrieval, pas de génération)
        ▼
  retrieve + generate    — chunks bruts, requête brute, aucun masquage
        │
        ▼
  B2  PIIAnalyzer + PIIAnonymizer (Presidio) — analyse PII de la RÉPONSE générée,
        │                                       puis masquage SÉLECTIF : une entité
        │                                       n'est masquée que si elle est jugée
        │                                       sensible, isolément ou en combinaison
        │                                       ré-identifiante ; les identifiants
        │                                       forts sont toujours masqués
        ▼
  B3  SADDetector (cascade F1→F2→F3) — sur la réponse déjà masquée : détecte une
                                        divulgation d'attribut sensible restante
                                        (regex → SBERT → Phi-3 Mini), et répare
                                        par reformulation, masquage de phrase,
                                        ou blocage complet en dernier recours
```
<img width="7040" height="2672" alt="cpsss" src="https://github.com/user-attachments/assets/0172fb50-5aa2-4a4b-932c-c56b09f4a0d5" />

Une brique **B0 (bootstrap)**, exécutée une seule fois à l'initialisation, auto-découvre le domaine du corpus (via `nvidia/domain-classifier`, repli Llama) et génère la taxonomie de catégories sensibles + les centroïdes SBERT utilisés par B1 et B3.

Il n'y a pas de brique de garde finale séparée (ResponseGuard) : B3 est le dernier filet.

`countermeasure_v6/` est **autonome** — il n'importe rien des dossiers `countermeasure*` antérieurs ; toutes les briques nécessaires sont dupliquées localement.

| Fichier | Rôle |
|---|---|
| `cpb_naive_rag_v6.py` | Orchestrateur du pipeline (B0→B3) autour de NaiveRAG/HybridRAG |
| `cpb_bootstrap_v6.py` | B0 — détection de domaine + génération de taxonomie |
| `cpb_query_risk_v6.py` | B1 — score de risque de la requête |
| `cpb_pii_v6.py` | B1 (budget gate) + B2 — analyse/anonymisation PII (Presidio) |
| `cpb_sad_detector_v6.py` | B3 — détection de divulgation d'attribut sensible |
| `cpb_ablation_v6.py` | Switches d'ablation leave-one-out (`AblationConfigV6`) |
| `cpb_models_v6.py` | Dataclasses partagées (résultats, findings, décisions) |
| `run_metrics_by_query_type_v6.py` | Métriques ventilées par type de requête, vs. naive RAG / masquage total |

> Dans le code (`AblationConfigV6`), chaque sous-composant garde son propre flag (`b1_query_risk`, `b2_budget_gate`, `b3_pii_analyzer`, `b4_pii_anonymizer`, `b6_sad_detector`) ; ce document les regroupe en 4 blocs fonctionnels B0–B3 pour la lisibilité.

## Étude d'ablation (CEDH / ECHR)

Corpus : [`ildpil/text-anonymization-benchmark`](https://huggingface.co/datasets/ildpil/text-anonymization-benchmark) (jurisprudence de la Cour européenne des droits de l'homme, annotée PII).

`evaluation_cedh_ablation/run_ablation_cumulative_v6.py` mesure l'apport de chaque brique par une ablation **cumulative en sens inverse** : on part du pipeline complet et on désactive une brique de plus à chaque étape, en partant du filet de sortie (B3) et en remontant vers l'entrée (B1) :

| Variante (nom dans le code) | Briques désactivées |
|---|---|
| `full_pipeline_v6` | aucune (baseline) |
| `cum_b6` | B3 (SAD detector) |
| `cum_b6_b3b4` | B3 + B2 (analyse + anonymisation PII) |
| `cum_b6_b3b4_b1b2` | B3 + B2 + B1 (query risk scorer + budget gate) |

B0 (bootstrap) n'est jamais désactivé dans cette étude : le couper forcerait aussi B3 à s'éteindre (plus de taxonomie/centroïdes), ce qui ferait coïncider cette étape avec `cum_b6_b3b4_b1b2`.

**Métriques** (100 % locales, aucun appel API externe) :
- **PII** — taux de fuite PII (`metrics/pii_leakage.py`, vérité terrain issue des annotations du corpus)
- **QS** — score de qualité pondéré, **AR** — pertinence de la réponse, **RL** — ROUGE-L, **BF1** — BERTScore F1 (`metrics/response_quality.py`)

L'échantillon (300 requêtes stratifiées : normal / extraction directe / injection / DGEA / MIA) est mis en cache dans `data/cedh_eval_ablation/sampled_queries.json` pour que toutes les variantes évaluent exactement les mêmes requêtes.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Un LLM local via [Ollama](https://ollama.com/) est requis :

```bash
ollama pull llama3.1:8b
```

Couches PII optionnelles (activées automatiquement selon le domaine détecté par B0) :

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

## Lancer l'étude d'ablation

```bash
python evaluation_cedh_ablation/run_ablation_cumulative_v6.py \
    --retrieval hybrid \
    --n-queries 300 \
    --seed 42
```

Options principales :
- `--variants full_pipeline_v6 cum_b6 ...` — n'exécuter qu'un sous-ensemble des variantes
- `--skip-generation` — réutiliser les réponses/chunks déjà générés pour une variante
- `--retrieval {hybrid,dense}` — HybridRAG (ChromaDB + BM25, fusion RRF) par défaut, ou NaiveRAG dense seul

Les résultats sont écrits dans `evaluation_cedh_ablation/cumulative_results_v6_<retrieval>/<variante>/`, avec un `summary.csv` récapitulatif par variante, et loggés dans MLflow (`mlruns/`, expérience `cedh_evaluation_ablation_cumulative_v6`).

## Structure du dépôt

```
countermeasure_v6/          contre-mesure CPB v6 (autonome)
evaluation_cedh_ablation/   étude d'ablation v6 sur le corpus CEDH
rag/                        architectures RAG (Naive, Self-RAG, hybride BM25+dense, GraphRAG)
attacks/                    attaques évaluées (extraction, injection, jailbreak, membership inference)
llms/                       backends LLM (Llama/Mistral locaux via Ollama, GPT-4o-mini, Claude Haiku)
metrics/                    calcul des métriques (fuite PII, qualité de réponse, score de vulnérabilité)
analysis/                   tracking MLflow, dashboard, heatmaps, tests statistiques (Wilcoxon)
knowledge_graph/, embeddings/, vectorstore/   infrastructure RAG partagée
data/                       jeux de données et échantillons
```

## Licence

MIT — voir [LICENSE](LICENSE).
