# Chatbot GraphRAG -- Tabagisme en Tunisie

Chatbot médical sourcé sur le tabagisme en Tunisie (législation, épidémiologie,
sevrage/TSN, dépendance, santé), construit sur une base de connaissances
Neo4j et un pipeline GraphRAG (recherche hybride vecteur + graphe). Il répond
de façon prudente et sourcée, sans diagnostic ni prescription, et s'abstient
plutôt que d'inventer une information absente du corpus.

Projet réalisé dans le cadre d'un stage de recherche (laboratoire LR01ES04).

> **Note sur le corpus** : les présentations sources du laboratoire
> (`documents/*.pptx`) et leur extraction texte (`textes/CHATBOT *.txt`,
> 4 fichiers sur 10) ne sont pas incluses dans ce dépôt public (diffusion
> non autorisée). Les 6 autres fichiers de `textes/` (sources publiques
> OMS, NHS, CDC, Santé publique France) sont versionnés normalement. Le
> graphe de connaissances ne peut donc pas être entièrement reconstruit
> depuis ce dépôt seul ; le jeu de test et les résultats d'évaluation
> (qui citent des faits, pas le contenu brut des diapositives) restent
> inclus en totalité.

## Architecture

```
documents/*.pptx  --(extract_documents.py)-->  textes/*.txt
                                                  |
                                   (build_knowledge_graph.py : SimpleKGPipeline
                                    de neo4j_graphrag, extraction via
                                    qwen2.5:7b local, schema/prompt dans
                                    kg_config.py, briques partagees avec
                                    reprocess_files.py dans
                                    kg_pipeline_common.py)
                                                  v
                                   Graphe de connaissances (Neo4j Aura)
                                                  |
                      (chatbot_pipeline.py : pipeline LangGraph --
                       routage securite -> reecriture -> recherche hybride
                       vecteur+Cypher -> grading LLM -> generation ->
                       verification groundedness. Generation via Groq
                       gpt-oss-120b/20b, embeddings nomic-embed-text local)
                                                  v
                                   app.py (interface web Streamlit)
```

`evaluation/` contient le jeu de test (`gold_eval_set.json`, 81 questions
vérifiées contre `textes/`), le script d'évaluation (`evaluate.py`), une
baseline RAG simple de comparaison (`baseline.py`) et une étude d'ablation
du canal graphe (`ablation_graphe.py`). Voir
`evaluation/rapport_evaluation.md` pour les résultats détaillés et
`docs/JOURNAL_TRAVAIL.md` pour l'historique complet des décisions prises
(journal de travail interne, non versionné -- voir `docs/JOURNAL_PUBLIC.md`
pour une version publique épurée).

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
- (Optionnel, pour `extract_documents.py` uniquement) [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  et [LibreOffice](https://www.libreoffice.org/) -- OCR et conversion
  WMF/EMF des images imbriquées dans les PPTX

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

Voir `.env.example` pour le modèle complet, y compris les 2 variables
optionnelles (`TESSERACT_CMD`, `SOFFICE_PATH`) utilisées uniquement par
`extract_documents.py` si Tesseract/LibreOffice ne sont pas installés aux
emplacements Windows par défaut.

## Utilisation

Dans l'ordre, pour (re)construire le graphe et lancer le chatbot :

1. **Extraction des sources** (si on repart des `.pptx`) :
   `python extract_documents.py` -- produit `textes/*.txt`.
2. **Construction du graphe** (graphe vide requis) :
   `python build_knowledge_graph.py` -- environ 1h-1h15 sur CPU, forte
   consommation RAM (Ollama + embeddings + driver Neo4j simultanés). En cas
   de panne mémoire en cours de route, utiliser `reprocess_files.py`
   pour traiter les fichiers un par un plutôt qu'en un seul run.
3. **Vider/sauvegarder le graphe si besoin** : `python scripts/wipe_graph.py`
   (sauvegarde horodatée dans un dossier `backup_wipe_*` puis vidage) ou
   `python scripts/backup_graph.py` (sauvegarde seule).
4. **Chatbot en ligne de commande** : `python chatbot_pipeline.py`.
5. **Interface web** : `streamlit run app.py` (réutilise
   directement la logique de `chatbot_pipeline.py`, rien à dupliquer).
   Couleur d'accent et police configurées dans `.streamlit/config.toml`.
   Icônes de thèmes : [Lucide](https://lucide.dev) (licence ISC), SVG
   recolorées à l'accent, intégrées inline dans `app.py` -- aucune
   dépendance externe au runtime.

   **Mode démo** (expose le détail technique du pipeline -- routage,
   recherche, notation des preuves -- utile pour une soutenance) : coché
   via la case "Mode démo" dans la barre latérale, ou en ouvrant l'URL
   avec `?debug=1` (ex. `http://localhost:8501/?debug=1`). Caché par
   défaut : un utilisateur normal ne voit jamais ce jargon. Si Groq
   répond 429 (quota atteint), même en mode démo l'app n'affiche jamais
   l'erreur brute de l'API (qui peut contenir un identifiant
   d'organisation) -- seulement "Quota Groq atteint (429)".

Pour vérifier rapidement que tout est opérationnel (Ollama/Neo4j/Groq +
pipeline sur 2 questions, ~30s) sans lancer les 81 questions du jeu de
test : `python smoke_test.py`.

## Tests unitaires

`smoke_test.py` vérifie que les services externes réels (Ollama, Neo4j,
Groq) répondent *maintenant*. `tests/` est complémentaire : il vérifie
que la *logique* du code (parsing des réponses LLM, routage par
mots-clés, filets de sécurité, scoring de l'évaluation) reste correcte à
chaque changement, sans dépendre d'aucun service externe ni d'un `.env`
présent -- utile en particulier vu l'historique d'incidents Aura/quota
Groq documenté dans `docs/JOURNAL_TRAVAIL.md`.

```bash
pip install -r requirements-dev.txt
pytest
```

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
avant la fin (voir `docs/JOURNAL_TRAVAIL.md` pour les incidents déjà rencontrés
et comment les diagnostiquer : toujours vérifier `correctness_score`/
`faithfulness_score` non-`None` avant de faire confiance aux chiffres
d'un run qui a touché un mur de quota).

Pour régénérer uniquement `rapport_evaluation.md` (+ le graphique) à partir
d'un résultat officiel déjà versionné, sans dépendre de `resultats_partial.json`
(local, non versionné, absent d'un clone frais) ni avoir besoin d'un accès à
Neo4j/Groq/Ollama (le chargement du pipeline est différé et jamais déclenché
dans ce mode) :

```bash
python evaluate.py --finalize-only --from resultats_20261011.json
```

`--from` accepte aussi un chemin relatif à `evaluation/`, par exemple
`--from historique/resultats_20261009.json`.

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
d'ablation du canal graphe : `evaluation/rapport_evaluation.md`. Les
résultats bruts des runs précédents (avant le rebuild et le correctif de
troncature du 2026-10-09) sont conservés à titre historique dans
`evaluation/historique/`.

## Limites connues (non corrigées à ce jour)

- Quelques vraies lacunes de retrieval (chunk attendu non trouvé par aucun
  canal) sur certaines questions de sevrage/dépendance -- notamment
  lorsqu'un fait précis est noyé dans un chunk centré sur un autre thème
  (listes denses, ex. Q19/Q47/Q57).
- Confusion occasionnelle entre le PNRT tunisien et son équivalent français
  dans les réponses sur des questions pièges.
- Ablations "sans grading"/"sans vérification de groundedness" conçues mais
  pas encore exécutées.
- Extraction instable du label `Programme` vs `Mesure` pour "Programme
  national de lutte anti-tabac" par le LLM d'extraction (qwen2.5:7b) --
  corrigé manuellement sur le graphe actuel, renforcé dans `kg_config.py`
  mais pas revalidé par un retraitement réel.

Voir `docs/JOURNAL_TRAVAIL.md` pour le détail diagnostique de chaque point.

## Structure du dépôt

```
extract_documents.py       Extraction .pptx/.pdf -> .txt
build_knowledge_graph.py   Construction du graphe de connaissances
chatbot_pipeline.py        Pipeline du chatbot (LangGraph)
reprocess_files.py         Retraitement d'un sous-ensemble de fichiers
kg_config.py                Schema et prompts d'extraction
kg_pipeline_common.py      Briques partagees build_knowledge_graph.py / reprocess_files.py
app.py                      Interface web (Streamlit) -- icônes Lucide inline, pas d'assets/
sources_meta.json          Nom affiché/organisme/URL par fichier source (app.py + chatbot_pipeline.py)
.streamlit/config.toml     Thème (couleur d'accent, police) de l'interface
smoke_test.py               Vérification rapide des services + du pipeline
scripts/                    Scripts ponctuels (sauvegarde, vidage, ajout manuel)
    backup_graph.py
    wipe_graph.py
    add_fctc_programme.py
poster/                     Poster (#BetterPoster) et ses figures
    poster_tabac.tex / poster_tabac.pdf
docs/                       Journal de travail et version publique épurée
    JOURNAL_TRAVAIL.md (non versionné -- voir .gitignore)
    JOURNAL_PUBLIC.md
documents/                   Sources .pptx d'origine
textes/                      Corpus texte extrait (entrée du build du graphe)
evaluation/                  Jeu de test, scripts et résultats d'évaluation
    historique/              Résultats des runs pré-11/10 (rebuild, pré-correctif)
tests/                       Tests unitaires (logique pure, sans service externe)
archive/                     Logs/sauvegardes de sessions passées (non versionné)
```
