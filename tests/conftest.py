"""
Fixtures partagees de la suite de tests unitaires.

Objectif : pouvoir importer 3_chatbot_ULTIME.py et tester ses fonctions
PURES (parsing, regex, routage par mots-cles...) sans jamais toucher un
service externe reel (Neo4j Aura, Groq, Ollama) -- contrairement a
smoke_test.py, qui lui verifie volontairement les vrais services. Les deux
sont complementaires : smoke_test.py = "l'environnement reel marche-t-il
maintenant", ce dossier = "la logique du code est-elle correcte, a chaque
commit, sans dependre d'Aura/Groq/Ollama disponibles ou d'un .env present".

Le module 3_chatbot_ULTIME.py construit un VectorCypherRetriever au niveau
module (ligne ~194), dont le constructeur interroge reellement Neo4j
(`SHOW VECTOR INDEXES`, voir neo4j_graphrag.retrievers.base.Retriever.
_fetch_index_infos) -- verifie en inspectant le code source de la lib.
Sans ce filet, importer le module planterait au moindre souci Aura (deja
arrive plusieurs fois, voir JOURNAL_TRAVAIL.md) ou sur une machine fraiche
sans .env (GROQ_API_KEY absent -> OpenAI() leve OpenAIError a la
construction, verifie aussi directement). On neutralise donc les DEUX
points d'E/S reseau au moment de l'import, sans modifier le fichier
source : le reste du fichier (regex, parsing, logique de routage) tourne
alors exactement comme en production.
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent


class _StubRetriever:
    """Remplace VectorCypherRetriever : memes attributs attendus par le
    reste du fichier, mais aucune requete Neo4j a la construction."""

    def __init__(self, *args, **kwargs):
        pass

    def search(self, *args, **kwargs):
        raise AssertionError(
            "La recherche vectorielle reelle ne doit jamais etre appelee "
            "dans la suite de tests unitaires (hermetique, sans service "
            "externe) -- si un test en a besoin, c'est un test d'integration, "
            "pas un test unitaire."
        )


@pytest.fixture(scope="session")
def chatbot():
    """Charge 3_chatbot_ULTIME.py une seule fois pour toute la session de
    test, avec Neo4j/Groq neutralises au moment de l'import."""
    import os
    import neo4j
    import neo4j_graphrag.retrievers as retrievers_module

    for var, dummy in [
        ("NEO4J_URI", "bolt://localhost:0"),
        ("NEO4J_USERNAME", "test"),
        ("NEO4J_PASSWORD", "test"),
        ("NEO4J_DATABASE", "test"),
        ("GROQ_API_KEY", "test-key-not-a-real-credential"),
    ]:
        os.environ[var] = dummy

    original_driver_fn = neo4j.GraphDatabase.driver
    original_retriever_cls = retrievers_module.VectorCypherRetriever
    neo4j.GraphDatabase.driver = staticmethod(lambda *a, **kw: MagicMock(name="fake_neo4j_driver"))
    retrievers_module.VectorCypherRetriever = _StubRetriever

    module_path = PROJECT_DIR / "3_chatbot_ULTIME.py"
    spec = importlib.util.spec_from_file_location("chatbot_ultime", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["chatbot_ultime"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        neo4j.GraphDatabase.driver = original_driver_fn
        retrievers_module.VectorCypherRetriever = original_retriever_cls

    return module
