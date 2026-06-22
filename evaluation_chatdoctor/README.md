# evaluation_chatdoctor

Application du protocole d'évaluation **Zhang et al. (IPM 2026)** au dataset médical
**ChatDoctor / HealthCareMagic** (`LinhDuong/chatdoctor-200k`), en miroir de
[`evaluation_zhang/`](../evaluation_zhang/) (corpus juridique australien).

Même méthode, même 6 métriques, même N=300 instances — seul le domaine change.
Toutes les données et caches sont isolés sous `data/chatdoctor_eval*` et
`data/chroma_chatdoctor*` : aucun fichier de `evaluation_zhang/` ou `data/zhang_eval/`
n'est lu ni écrit par ce dossier.

## Fichiers

| Fichier | Rôle |
|---|---|
| `dataset_prep.py` | Télécharge `LinhDuong/chatdoctor-200k`, échantillonne 300 dialogues (seed=42), construit `"Patient: ...\nDoctor: ..."`, chunke (500/50), embedde et met en cache (`chunks_cache.json` + `chunks_embeddings.npy`). Sort `data/chatdoctor_eval/doc_index.json` |
| `index_chunks.py` | Insertion ChromaDB isolée dans un process minimal (voir note ci-dessous). Resumable via `collection.count()`. Appelé automatiquement par `dataset_prep.py` |
| `attack_builder.py` | Reproduit exactement le protocole d'attaque Zhang (extraction GPT-4o → split known/privacy → query template) avec 6 attributs **médicaux** : `symptom`, `patient_profile`, `duration` (connus) / `diagnosis`, `medication`, `treatment` (cibles) |
| `run_evaluation_chat_doctor.py` | Orchestrateur principal : génère les réponses CPB v3 (llama3.1:8b), calcule LO/AE/PI/CR/SS/AR, exporte CSV + MLflow (`chatdoctor_evaluation`) |
| `test_pipeline_5docs.py` | Test rapide isolé sur 5 dialogues (`data/chatdoctor_eval_test/`) avant de lancer le run complet |

## Pourquoi ce n'est pas juste un import de `evaluation_zhang/`

`metric_pi.py` (dans `evaluation_zhang/`) code en dur ses chemins de cache vers
`data/zhang_eval/` et `data/chroma_zhang_claims/`. Le réutiliser directement aurait
soit pollué/écrasé les résultats Zhang existants, soit rechargé par erreur sa DB de
claims juridiques. `run_evaluation_chat_doctor.py` importe donc uniquement les
fonctions de **calcul pur** (`compute_lo`, `score_ae`, `PIMetric._decompose_*`,
`PIMetric._precompute_weights`, `PIMetric.compute_pi`) et gère lui-même tous les
chemins de cache. RAGAS (CR/SS/AR) est appelé directement (pas via
`metric_utility.compute_utility`) pour pouvoir le découper en lots cachés —
voir `compute_utility_chunked`.

Note : `PIMetric.compute_pi_batch()` contient un hook de debug temporaire
(`evaluation_zhang/metric_pi.py:378-382`) qui fait `sys.exit(0)` après la 1ère instance.
Ce dossier ne l'appelle jamais — il boucle sur `PIMetric.compute_pi()` (sans le hook)
instance par instance.

## Bug connu : crash natif ChromaDB (Windows)

Sur cette machine, `chromadb` 1.5.9 (backend Rust) plante de façon intermittente
(`Windows fatal exception: access violation` dans `chromadb/api/rust.py`, sur
`_add` **et** `_count`) au lieu de la version `0.5.0` prévue par
`requirements.txt`.

**Cause confirmée n°1 (root cause, reproduite de façon fiable) :** initialiser
`mlflow` (même un simple `mlflow.set_tracking_uri()` + `set_experiment()`, sans
`start_run()`) dans le **même process** qui touche ensuite ChromaDB fait planter
quasi systématiquement le premier appel ChromaDB qui suit (testé 4/4 dans cet
ordre = crash, 2/2 dans l'ordre inverse = succès). Probable conflit natif entre
le moteur Rust de ChromaDB et la pile SQLAlchemy/SQLite du `FileStore` mlflow.
**Fix : `run_evaluation_chat_doctor.py` n'importe `mlflow` qu'à la toute fin de
`main()`, après que tout le travail ChromaDB (génération CPB v3 + PI) soit
terminé** — jamais avant, jamais dans le même bloc.

**Cause probable n°2 (mitigée mais moins précisément isolée) :** les premiers
crashs rencontrés pendant `collection.add()` lors du chunking initial
(`dataset_prep.py`) n'impliquaient pas mlflow — vraisemblablement la même classe
d'instabilité Rust/Windows, déclenchée différemment. Mitigation : `dataset_prep.py`
calcule chunks + embeddings, les met en cache sur disque (`chunks_cache.json`,
`chunks_embeddings.npy`), puis délègue l'insertion à `index_chunks.py` — un
script minimal qui n'importe que `chromadb` + `numpy` — relancé automatiquement
(jusqu'à 30 fois) s'il crashe ; chaque tentative reprend depuis `collection.count()`.
`index_chunks.py` est générique (args CLI) et réutilisé pour la DB de claims PI
(`build_chatdoctor_claims_db`).

Si un nouveau crash ChromaDB apparaît ailleurs, vérifier en premier l'ordre
d'initialisation mlflow avant de chercher une autre cause.

## Reprise sur crash — chaque boucle payante est checkpointée

`run_evaluation_chat_doctor.py` peut tourner plusieurs heures et faire des milliers
d'appels GPT-4o (~3-7$ estimés sur 300 instances, RAGAS pouvant lui seul dépasser
1000 appels). Pour qu'un crash en cours de route (crash ChromaDB ci-dessus, coupure
réseau, rate limit OpenAI, Ctrl+C...) ne fasse perdre que l'appel en vol — jamais
toute une étape déjà payée — chaque boucle sur les 300 instances utilise
`run_checkpointed()` : le résultat est réécrit sur disque après **chaque** instance,
et relancer le script reprend automatiquement depuis `len(résultats déjà en cache)`.

Concerné : génération CPB v3 (`cpb_results.json`, gratuit mais coûte du temps),
AE (`ae_results.json`), décomposition PI par document (`claims_decomposed.json`),
scoring PI par réponse (`pi_scores.json`), réponses de référence
(`reference_responses.json`). RAGAS (`compute_utility_chunked`) est découpé en
lots de 25 instances (`utility_chunks/chunk_NNN.json`) plutôt que checkpointé
instance par instance, car `ragas.evaluate()` traite un lot entier en un seul
appel opaque — un crash ne fait perdre qu'un lot (25 instances) au lieu des 300.

## Usage

```bash
cd evaluation_chatdoctor

# 1. Test rapide (5 dialogues, peu de coût API)
python test_pipeline_5docs.py

# 2. Indexation des 300 dialogues (gratuit, pas d'appel LLM)
python dataset_prep.py

# 3. Run complet (300 instances, GPT-4o pour attaques/métriques + llama3.1:8b pour CPB v3)
python run_evaluation_chat_doctor.py
python run_evaluation_chat_doctor.py --skip-generation   # réutilise les réponses en cache
```

## Sorties

```
data/chatdoctor_eval/
  doc_index.json              # 300 dialogues {doc_id: {text, metadata}}
  attack_queries.json         # requêtes d'attaque (protocole Zhang, attributs médicaux)
  cpb_results.json            # checkpoint génération CPB v3 (response+contexts par instance)
  responses.json / contexts.json   # dérivés de cpb_results.json une fois complet
  ae_results.json             # checkpoint juge AE (par instance)
  claims_decomposed.json      # checkpoint décomposition PI par document (par instance)
  claims_cache.json / claims_embeddings.npy   # claims PI finalisées (post-décomposition)
  pi_scores.json               # checkpoint scoring PI (par instance)
  reference_responses.json    # checkpoint réponses de référence GPT-4o (par instance)
  utility_chunks/chunk_NNN.json   # checkpoint RAGAS (par lot de 25 instances)
  results.json                # résultats agrégés + per_instance
  results_per_query.csv       # une ligne par requête

data/chroma_chatdoctor/            # collection chatdoctor_eval_corpus (300 docs chunkés)
data/chroma_chatdoctor_claims/      # collection chatdoctor_claims (PI, claims atomiques)
```
