# Chatbot GraphRAG -- Tabagisme en Tunisie

Chatbot médical sourcé sur le tabagisme en Tunisie (législation, épidémiologie,
sevrage/TSN, dépendance, santé), construit sur une base de connaissances
Neo4j et un pipeline GraphRAG (recherche hybride vecteur + graphe). Il répond
de façon prudente et sourcée, sans diagnostic ni prescription, et s'abstient
plutôt que d'inventer une information absente du corpus.

Projet réalisé dans le cadre d'un stage de recherche (laboratoire LR01ES04).

## Architecture

```
documents/*.pptx  --(1_extract_pptx.py)-->  textes/*.txt
                                                  |
                                   (2_build_kg_ULTIME.py : SimpleKGPipeline
                                    de neo4j_graphrag, extraction via
                                    qwen2.5:7b local, schema/prompt dans
                                    kg_config.py)
                                                  v
                                   Graphe de connaissances (Neo4j Aura)
                                                  |
                      (3_chatbot_ULTIME.py : pipeline LangGraph --
                       routage securite -> reecriture -> recherche hybride
                       vecteur+Cypher -> grading LLM -> generation ->
                       verification groundedness. Generation via Groq
                       gpt-oss-120b/20b, embeddings nomic-embed-text local)
                                                  v
                                   app_streamlit.py (interface web)
```

`evaluation/` contient le jeu de test (`gold_eval_set.json`, 81 questions
vérifiées contre `textes/`), le script d'évaluation (`evaluate.py`), une
baseline RAG simple de comparaison (`baseline.py`) et une étude d'ablation
du canal graphe (`ablation_graphe.py`). Voir
`evaluation/rapport_evaluation.md` pour les résultats détaillés et
`JOURNAL_TRAVAIL.md` pour l'historique complet des décisions prises.

## Prérequis

- Python 3.12
- [Ollama](https://ollama.com) installé localement, avec les modèles :
  ```
  ollama pull qwen2.5:7b
  ollama pull nomic-embed-text
  ```
- Une instance [Neo4j Aura](https://console.neo4j.io) (ou Neo4j local) avec
  APOC activé (utilisé pour la déduplication par accents/casse)
- Une clé API [Groq](https://console.groq.com) (génération + juges LLM
  d'évaluation -- tier gratuit utilisable mais limité à 200 000
  tokens/jour sur `openai/gpt-oss-20b`)

## Installation

```bash
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine (jamais commité, voir `.gitignore`) :

```
NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io
NEO4J_USERNAME=<utilisateur>
NEO4J_PASSWORD=<mot_de_passe>
NEO4J_DATABASE=<nom_de_la_base>
GROQ_API_KEY=<cle_groq>
```

## Utilisation

Dans l'ordre, pour (re)construire le graphe et lancer le chatbot :

1. **Extraction des sources** (si on repart des `.pptx`) :
   `python 1_extract_pptx.py` -- produit `textes/*.txt`.
2. **Construction du graphe** (graphe vide requis) :
   `python 2_build_kg_ULTIME.py` -- environ 1h-1h15 sur CPU, forte
   consommation RAM (Ollama + embeddings + driver Neo4j simultanés). En cas
   de panne mémoire en cours de route, utiliser `retraiter_fichiers.py`
   pour traiter les fichiers un par un plutôt qu'en un seul run.
3. **Vider/sauvegarder le graphe si besoin** : `python wipe_graph.py`
   (sauvegarde horodatée dans un dossier `backup_wipe_*` puis vidage) ou
   `python backup_graph.py` (sauvegarde seule).
4. **Chatbot en ligne de commande** : `python 3_chatbot_ULTIME.py`.
5. **Interface web** : `streamlit run app_streamlit.py` (réutilise
   directement la logique de `3_chatbot_ULTIME.py`, rien à dupliquer).

Pour vérifier rapidement que tout est opérationnel (Ollama/Neo4j/Groq +
pipeline sur 2 questions, ~30s) sans lancer les 81 questions du jeu de
test : `python smoke_test.py`.

## Évaluation

```bash
cd evaluation
python evaluate.py          # pipeline complet sur les 81 questions
python baseline.py          # baseline RAG simple de comparaison
python ablation_graphe.py --config B --categories all   # sans le canal graphe
python ablation_graphe.py --config C --categories all   # avec le canal graphe
python ablation_graphe.py --finalize --categories all   # tableau comparatif
```

`evaluate.py` sauvegarde après chaque question (`resultats_partial.json`,
non versionné) et reprend automatiquement en cas d'interruption -- utile
car le quota Groq gratuit (TPD) ou la RAM disponible peuvent couper un run
avant la fin (voir `JOURNAL_TRAVAIL.md` pour les incidents déjà rencontrés
et comment les diagnostiquer : toujours vérifier `correctness_score`/
`faithfulness_score` non-`None` avant de faire confiance aux chiffres
d'un run qui a touché un mur de quota).

### Résultats clés (run du 2026-10-11, 81/81)

| Métrique | Valeur |
|---|---|
| Précision du routage | 1.0 |
| Précision du comportement | 0.901 |
| Rappel éléments attendus | 0.554 |
| Faithfulness moyenne (/10) | 10.0 |
| Correctness moyenne (/5) | 4.215 |
| Hit-rate retrieval hybride | 0.836 |

Détail complet, comparaison avec la baseline RAG simple et étude
d'ablation du canal graphe : `evaluation/rapport_evaluation.md`.

## Limites connues (non corrigées à ce jour)

- Quelques vraies lacunes de retrieval (chunk attendu non trouvé par aucun
  canal) sur certaines questions de sevrage/dépendance.
- Confusion occasionnelle entre le PNRT tunisien et son équivalent français
  dans les réponses sur des questions pièges.
- Ablations "sans grading"/"sans vérification de groundedness" conçues mais
  pas encore exécutées.
- Extraction instable du label `Programme` vs `Mesure` pour "Programme
  national de lutte anti-tabac" par le LLM d'extraction (qwen2.5:7b) --
  corrigé manuellement sur le graphe actuel, pas à la racine du prompt.

Voir `JOURNAL_TRAVAIL.md` pour le détail diagnostique de chaque point.

## Structure du dépôt

```
1_extract_pptx.py          Extraction .pptx -> .txt
2_build_kg_ULTIME.py       Construction du graphe de connaissances
3_chatbot_ULTIME.py        Pipeline du chatbot (LangGraph)
kg_config.py                Schema et prompts d'extraction
retraiter_fichiers.py      Retraitement d'un sous-ensemble de fichiers
backup_graph.py / wipe_graph.py   Sauvegarde / vidage du graphe
ajouter_fctc_2030.py       Script d'ajout ponctuel (entité FCTC 20-30)
app_streamlit.py           Interface web
assets/                    Icônes de l'interface
documents/                 Sources .pptx d'origine
textes/                    Corpus texte extrait (entrée du build du graphe)
evaluation/                 Jeu de test, scripts et résultats d'évaluation
archive/                   Logs/sauvegardes de sessions passées (non versionné)
```
