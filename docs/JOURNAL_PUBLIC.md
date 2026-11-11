# Journal de travail -- version publique

Version épurée de `docs/JOURNAL_TRAVAIL.md` (non versionné -- voir `.gitignore`),
qui retrace l'historique complet de session. Cette version garde les diagnostics,
décisions techniques et résultats, et retire les détails de session sans intérêt
pour un lecteur externe (incidents de quota/RAM résolus par simple reprise,
contexte personnel, identifiants d'instance).

## Audit initial et corrections de routage (2026-10-07)

Le projet existant au démarrage de cette phase de travail différait du brief de
mission reçu : un jeu de test `gold_eval_set.json` (20 questions) et un script
`4_evaluate_chatbot.py` existaient déjà sur le disque, visiblement issus d'un
travail antérieur. Décision : partir de cet état réel plutôt que de recréer un
système d'évaluation parallèle, et l'étendre plutôt que le remplacer.

**Deux échecs connus corrigés** (`chatbot_pipeline.py`) :
- *Demi-vie de la nicotine* : le label `Substance` existait dans le graphe mais
  n'était tout simplement pas routé par `INTENT_PRIORITY` -- la question passait
  donc uniquement par la recherche vectorielle, qui ne remontait pas le bon chunk
  dans son top-8. Ajout d'une entrée `Substance` dans `INTENT_PRIORITY` et
  généralisation du filtrage par nom (jusque-là réservé à `Produit`) à
  n'importe quel label.
- *Coût économique du tabagisme* : même cause -- `Pays`/`Statistique` existaient
  dans le graphe sans entrée de routage dédiée. Corrigé de la même façon.
- Bug additionnel trouvé en chemin : `detect_intent` ne retournait que la
  PREMIÈRE catégorie qui correspondait, perdant les autres catégories
  pertinentes pour une question à cheval sur deux sujets. Corrigé pour retourner
  la liste complète des catégories qui matchent, et cumuler les candidats de
  chaque label détecté.
- Bug additionnel : le filtre par nom ne gérait pas les accents (`toLower()` ne
  les retire pas en Cypher) -- corrigé via `apoc.text.clean()`.

## Audit du graphe (2026-10-07)

1. **Doublons par accents non fusionnés** : 20 groupes de doublons purs détectés
   (ex. "Etats-unis"/"États-Unis") -- la fusion automatique existante ne
   normalisait que la casse, pas les accents. Corrigé par une clé de regroupement
   basée sur `apoc.text.clean()`, avec préférence pour le nom accentué comme
   survivant. Sauvegarde complète du graphe faite avant toute fusion.
2. **Collision de label avec "Source"** : le schéma JSON imposé au LLM
   d'extraction contraint la structure mais pas les valeurs de `label` --
   qwen2.5:7b produisait parfois des noeuds `Source` sans propriété `path` pour
   des références bibliographiques citées dans le texte. Ajout d'un type
   `Citation` dédié dans `kg_config.py`, et filet de sécurité déterministe qui
   relabellise après coup tout noeud `Source` sans `path`.
3. **Doublons de Source dus à une incohérence de séparateur de chemin** : sur
   Windows, `glob.glob()` renvoie des chemins avec antislash, utilisés comme clé
   d'unicité pour le noeud Source -- un script utilisant des chemins écrits en
   dur avec slash ne retrouvait donc pas le noeud existant et en créait un
   second. Corrigé par normalisation systématique en slash partout où le chemin
   sert de clé.

## Système d'évaluation (2026-10-07/08)

Jeu de test étendu à **81 questions** (`evaluation/gold_eval_set.json`), toutes
vérifiées à la main contre le corpus. Catégories : législation, épidémiologie,
sevrage, dépendance, santé, sécurité (urgence/demande personnelle/hors domaine).
6 questions-pièges explicites (info absente du corpus, pays différent, etc.).

- `evaluation/common.py` : scoring déterministe du rappel d'éléments,
  hit-rate/MRR de retrieval, agrégation par catégorie/type, rapport Markdown.
- `evaluation/evaluate.py` : fait tourner le pipeline complet sur les 81
  questions. Juge LLM (faithfulness + correctness) sur un modèle différent de
  celui qui génère la réponse, pour éviter qu'un modèle se note lui-même.
- `evaluation/baseline.py` : RAG simple de comparaison (vecteur seul, un seul
  appel LLM, aucun routage/grading/groundedness) -- pour chiffrer l'apport réel
  du pipeline contrôlé.
- Retries avec backoff exponentiel ajoutés sur les erreurs 429 (rate limit) du
  tier gratuit Groq, actifs pour le pipeline et les scripts d'évaluation.

## Rebuild complet du graphe (2026-10-08/09)

Le rebuild complet (10 documents, ~1h-1h15 sur CPU) s'est heurté à plusieurs
interruptions liées à la consommation mémoire élevée du pipeline (Ollama local +
embeddings + driver Neo4j simultanés) et, séparément, au quota Groq gratuit
pendant les runs d'évaluation qui ont suivi. Chaque interruption a été gérée par
sauvegarde systématique avant toute opération destructive (vidage du graphe) et
reprise automatique (checkpoint après chaque question/fichier) -- jamais de
relance automatique après un incident sans vérification préalable des services.
Face aux interruptions répétées, le traitement a basculé sur un mode fichier par
fichier (`reprocess_files.py`, mémoire remise à zéro entre chaque fichier)
plutôt qu'un unique run sur les 10 documents.

**Contrôles qualité sur le graphe final** (706 noeuds, 10 Source) : 0 doublon de
casse/accents, 0 doublon inter-types, 0 entité orpheline, 0 noeud `Source` sans
`path`. Un bug résiduel (le PNRT extrait avec le label `Mesure` au lieu de
`Programme`) corrigé manuellement sur le graphe vivant, et renforcé dans le
prompt de `kg_config.py` pour les futurs rebuilds (non revalidé par un
retraitement réel).

### Résultats avant/après rebuild (81/81)

| Métrique | Avant rebuild (08/10) | Après rebuild (09/10) |
|---|---|---|
| Précision du routage | 0.988 | 1.0 |
| Rappel éléments attendus | 0.327 | 0.473 |
| Correctness moyenne (/5) | 3.05 | 3.967 |
| Latence moy. (s) | 90.97 | 40.66 |

## Bug racine trouvé : troncature du grading à 700 caractères (2026-10-09)

Diagnostic sur une question dont le bon chunk était bien retrouvé
(`hit_hybrid=True`) mais dont aucune preuve n'était retenue après grading
(`n_evidence_retained=0`) : le fait recherché se trouvait à la position
1407/1776 caractères du chunk, alors que `grade_evidence()` ne montrait au LLM
de grading que les 700 premiers caractères de chaque candidat -- le fait était
donc structurellement invisible au moment du jugement de pertinence. Le LLM
rejetait "à raison" compte tenu de ce qu'il voyait réellement.

**Corrigé** : troncature relevée de 700 à 1500 caractères pour le grading (le
point le plus sensible côté quota Groq, mais la cause racine réelle), et à 2000
pour la génération et la vérification de groundedness (qui ne portent que sur
un petit nombre de preuves déjà filtrées -- coût marginal). Vérifié en direct
sur plusieurs questions concernées : réponse correcte et sourcée après le
correctif, alors qu'elles faisaient toutes partie des abstentions injustifiées
du rapport précédent.

**Découverte en chemin** : deux limites Groq distinctes à surveiller
séparément -- TPD (tokens par jour, 200 000) et TPM (tokens par minute, 8 000,
qui peut être déclenchée même avec un quota journalier intact si un seul appel
envoie trop de candidats à la fois). Toute augmentation future de
`TOP_K_VECTOR`/`MAX_GRAPH_CANDIDATES` doit être testée avec cette limite TPM en
tête.

## Résultats finaux définitifs (81/81, 2026-10-11, après nettoyage complet)

Un nettoyage plus rigoureux des runs partiels (filtrage sur `faithfulness_score`
ET `correctness_score`, pas seulement l'un des deux -- une leçon méthodologique
tirée en cours de route) a permis d'obtenir des chiffres finaux propres :

| Métrique | Valeur |
|---|---|
| Précision du routage | 1.0 |
| Précision du comportement | 0.901 |
| Rappel éléments attendus | 0.554 |
| Faithfulness moyenne (/10) | 10.0 |
| Correctness moyenne (/5) | 4.215 |
| Hit-rate hybride | 0.836 |
| Latence moy./méd. (s) | 32.23 / 30.64 |

### Lacunes de retrieval réexaminées

Plusieurs questions historiquement flaguées "lacune de retrieval" se sont
révélées être un biais de la méthode d'évaluation (le `expected_snippet` choisi
à la main ne correspondait pas exactement au chunk réellement utilisé par le
chatbot, alors que la réponse était correcte) plutôt que de vraies lacunes.
Trois cas restent de vraies lacunes, toutes de la même famille -- une
information précise noyée dans un chunk plus large centré sur un autre thème
(listes denses de médicaments, de dosages, de statistiques) :
- Un cas de réponse donnée avec confiance mais avec des chiffres erronés (le
  cas le plus préoccupant pour un chatbot médical -- une statistique proche
  mais différente était remontée en premier par la recherche vectorielle).
- Deux cas d'abstention correcte mais de rappel manquant.

Investigation d'une piste de correctif (augmenter `TOP_K_VECTOR`) abandonnée :
testée isolément, elle déclenche la limite TPM de Groq (voir plus haut) --
augmenter ces seuils globalement n'est donc pas un changement sans risque.
Un vrai correctif demanderait de revoir la granularité du chunking pour les
listes denses, pas un ajustement de seuil.

## Ablation du canal graphe (2026-10-09/10)

Comparaison de 3 configurations sur les 81 questions : RAG simple (vecteur
seul), pipeline complet sans le canal graphe Cypher, pipeline complet avec.

| Métrique | Sans graphe | Avec graphe |
|---|---|---|
| Précision routage | 0.938 | **1.0** |
| Rappel éléments | 0.542 | 0.569 |
| Justesse (/5) | 4.292 | 4.338 |
| Taux abstention | 0.222 | **0.136** |

Lecture honnête : le gain de rappel brut du canal graphe est modeste sur
l'ensemble, concentré sur les catégories où le graphe contient des entités
structurées (législation, dépendance, santé) et absent sur d'autres
(épidémiologie). Le gain le plus net et le plus fiable est ailleurs :
précision du routage et surtout taux d'abstention nettement plus bas -- le
canal graphe fournit des preuves structurées que le filtrage de pertinence
retient plus facilement, réduisant les abstentions injustifiées.

## Mise en ordre du projet (2026-10-11)

Structuration en 8 étapes : versioning git, hygiène des secrets (clés Groq
révoquées après usage), nettoyage/archivage des fichiers de debug, figeage des
dépendances (`requirements.txt`), documentation (`README.md`), centralisation
de la configuration (constantes de troncature nommées plutôt que dupliquées à
trois endroits -- exactement le genre de duplication qui avait causé le bug de
troncature à 700 décrit plus haut), script de vérification rapide
(`smoke_test.py`), intégration durable de la section d'ablation dans le
générateur de rapport (pour qu'elle survive aux régénérations automatiques).

Suite de tests unitaires ajoutée (71 tests, `tests/`) sur la logique pure du
chatbot (parsing, routage par mots-clés, filets de sécurité) et du scoring
d'évaluation, sans dépendre d'aucun service externe.

## Préparation pour publication (2026-10-12)

Identifiant réel de l'instance Neo4j et présentations du laboratoire purgés de
tout l'historique git avant toute publication éventuelle. Les clés Groq
utilisées pendant le développement ont été révoquées après usage.

## Refonte du poster (2026-10-14/15)

Première version (mise à jour des chiffres, format académique classique)
rejetée après retour. Recherche sur le design de poster académique moderne
avant de recommencer : format **#BetterPoster** choisi (message central +
colonne "silent presenter" + colonne "ammo bar"), avec vérification par
compilation LaTeX réelle et rasterisation (la session précédente n'avait livré
aucun poster jamais vérifié visuellement).

## Nettoyage et restructuration du dépôt (2026-11-11)

Voir l'entrée correspondante dans `docs/JOURNAL_TRAVAIL.md` (renommage des
scripts, factorisation du pipeline KG, correction de messages/commentaires
périmés, séparation du journal complet non-publiable de cette version
publique). Ajout notable pour la reproductibilité : `evaluate.py --finalize-only
--from resultats_20261011.json` régénère `rapport_evaluation.md` à partir d'un
résultat officiel versionné, sans dépendre du checkpoint local
`resultats_partial.json` (non versionné) ni nécessiter d'accès à Neo4j/Groq/Ollama.

## Limites connues restantes

- Chunking des listes denses (médicaments, dosages, statistiques) : cause de
  plusieurs lacunes de retrieval, pas corrigé par prudence (risque de
  régression plus large que le gain sur un cas isolé).
- Ablations "sans grading"/"sans vérification de groundedness" conçues mais
  jamais exécutées.
- Confusion occasionnelle entre le PNRT tunisien et son équivalent français sur
  des questions pièges.
- Bug d'extraction `Mesure`/`Programme` renforcé au niveau du prompt mais pas
  revalidé par un retraitement réel.
