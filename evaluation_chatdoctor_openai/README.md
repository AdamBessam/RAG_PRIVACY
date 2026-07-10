# evaluation_chatdoctor_openai

Évaluation Zhang et al. sur le dataset médical **ChatDoctor / HealthCareMagic**,
dans les **mêmes conditions que le run publié** `evaluation_zhang_openai/run_evaluation_openai_v2_hybrid_nodedup.py`,
mais avec la contre-mesure **CPB v5 combo** et le **bloc B7 désactivé**.

## Ce qui est testé

| Dimension | Valeur |
|---|---|
| Retrieval | hybride dense (`text-embedding-3-small`) + BM25, fusion RRF, `dedup=False` |
| Génération | `gpt-4o-mini` |
| Juges (AE / PI / RAGAS) | `gpt-4o` |
| Utilité | RAGAS **context_recall** (= « Context Recall » de Zhang Table 2), SS, AR |
| Contre-mesure | **CPB v5 combo** (masquage par combinaisons ré-identifiantes domain-aware) |
| Ablation | **B7 (`CPBResponseGuard`) désactivé** — `AblationConfig(b7_response_guard=False)` |

Comparaison de référence : le combo *complet* (B7 on) et la baseline v4 (masque-tout).

## Lancer

```bash
python evaluation_chatdoctor_openai/run_evaluation_chatdoctor_combo_b7off.py
# reprise sans régénérer les réponses :
python evaluation_chatdoctor_openai/run_evaluation_chatdoctor_combo_b7off.py --skip-generation
```

Prérequis machine : clé OpenAI (`config.OPENAI_API_KEY`), `rank_bm25`, `ragas==0.1.21`,
`langchain-openai`, et le modèle local de B0 (bootstrap domaine) déjà présent pour CPB.
Le run est **checkpointé** (génération, AE, PI, références, RAGAS) : un crash reprend
où il s'était arrêté.

## Sorties (dans `data/chatdoctor_eval_openai_combo_b7off/`)

| Fichier | Contenu |
|---|---|
| `responses.json` | réponses finales du système protégé (300) |
| `reference_responses.json` | réponses gold de référence (copie locale) |
| `results_per_query.csv` | 1 ligne/requête : `query`, `reference`, `response`, LO, AE, PI |
| `exemples_questions_reponses.md` | lisible : Question / Réponse de référence / Réponse du système |
| `results.json` | métriques agrégées + `responses` + `reference_responses` + per-instance |
| `contexts.json` | chunks bruts récupérés par requête |

## Métriques (rappel des directions)

| | Sens | Description |
|---|---|---|
| LO_F1 | ↓ | ROUGE-L réponse vs document source (fuite verbatim) |
| AE | ↑ | juge GPT-4o 1/2/3 (3 = ne révèle rien) |
| PI | ↓ | ré-identification de l'individu source (top-5 sur base de claims) |
| CR | ↑ | context_recall (qualité du retrieval) |
| SS | ↑ | similarité réponse vs réponse gold |
| AR | ↑ | pertinence réponse vs question |

## Isolation / rollback

Tout est **isolé**, rien d'existant n'est écrasé :

- caches variante → `data/chatdoctor_eval_openai_combo_b7off/`
- corpus médical ré-embeddé OpenAI → `data/chroma_chatdoctor_openai/` (collection `chatdoctor_eval_corpus_openai`, auto-construite au 1ᵉʳ run)
- caches **partagés** réutilisés (indépendants de la variante, payés une fois) :
  base de claims PI médicale (`data/chroma_chatdoctor_claims`) et réponses de
  référence gold (`data/chatdoctor_eval/reference_responses.json`), via les helpers
  éprouvés de `evaluation_chatdoctor/run_evaluation_chat_doctor.py`.

Rollback = supprimer ce dossier + `data/chatdoctor_eval_openai_combo_b7off/`
+ `data/chroma_chatdoctor_openai/`. `countermeasure_v5/` n'est pas modifié.

## Note technique (combo + gpt-4o-mini)

`CPBNaiveRAGV5Combo._llama_json` suppose un client **ollama**
(`self.llm.client.chat(..., format="json")`). Avec `GPT4oMiniLLM`, `self.llm.client`
est un `openai.OpenAI` dont `.chat` n'est pas appelable → la découverte des
combinaisons échouerait et le combo retomberait silencieusement en v5 (aucune
combinaison). La sous-classe locale `CPBComboOpenAI` override `_llama_json` pour
passer par `self.llm.generate()` (voie OpenAI ; le prompt force déjà « valid JSON only »).
