"""
============================================================
smoke_test.py -- VERIFICATION RAPIDE DU PIPELINE
PROJET TABAC TUNISIE
============================================================
Verifie en quelques dizaines de secondes, sans lancer les 81 questions de
evaluation/gold_eval_set.json, que les 3 services externes repondent et que
le pipeline produit une reponse coherente sur 2 questions (1 routage
special, 1 question factuelle complete). Objectif : detecter une
regression (code, service, quota) sans attendre un run complet ni gaspiller
le quota Groq journalier dessus.

Usage : python smoke_test.py (depuis la racine du projet)
Code de sortie : 0 si tout passe, 1 sinon.
"""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# 3_chatbot_ULTIME.py imprime des emojis -- sans ca, la console Windows
# (cp1252) plante avec un UnicodeEncodeError des le premier print (meme
# correctif que evaluation/evaluate.py).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

PROJECT_DIR = Path(__file__).parent
CHATBOT_MODULE_PATH = PROJECT_DIR / "3_chatbot_ULTIME.py"

results = []  # [(nom, ok: bool, detail: str, duree: float)]


def run_check(name, fn):
    start = time.time()
    try:
        detail = fn()
        results.append((name, True, detail or "OK", time.time() - start))
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}", time.time() - start))


def check_env():
    required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GROQ_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"manquantes : {', '.join(missing)}")
    return "toutes presentes"


def check_ollama():
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "commande 'ollama list' en echec")
    needed = ["qwen2.5:7b", "nomic-embed-text"]
    missing = [m for m in needed if m not in out.stdout]
    if missing:
        raise RuntimeError(f"modele(s) absent(s) : {', '.join(missing)} (ollama pull ...)")
    return "modeles presents"


def check_neo4j():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    try:
        with driver.session() as session:
            count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        if count == 0:
            raise RuntimeError("graphe vide (0 noeud) -- build necessaire avant usage")
        return f"{count} noeuds"
    finally:
        driver.close()


def check_groq():
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
    )
    if not resp.choices:
        raise RuntimeError("reponse vide")
    return "repond"


def print_results():
    print()
    for name, ok, detail, dur in results:
        icon = "OK  " if ok else "FAIL"
        print(f"[{icon}] {name:50s} ({dur:4.1f}s) -- {detail}")
    print()


def main():
    print("=" * 70)
    print("SMOKE TEST -- chatbot GraphRAG tabac Tunisie")
    print("=" * 70)

    run_check("Variables d'environnement (.env)", check_env)
    run_check("Ollama local (qwen2.5:7b + nomic-embed-text)", check_ollama)
    run_check("Neo4j Aura (connectivite + nombre de noeuds)", check_neo4j)
    run_check("Groq API (appel minimal)", check_groq)
    print_results()

    if any(not ok for _, ok, _, _ in results):
        print("Un prerequis de base a echoue -- pipeline non teste (evite de")
        print("gaspiller du quota Groq sur un environnement deja casse).")
        sys.exit(1)

    print("Prerequis OK. Test du pipeline complet sur 2 questions...\n")

    spec = importlib.util.spec_from_file_location("chatbot_ultime", CHATBOT_MODULE_PATH)
    chatbot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chatbot)
    app = chatbot.build_graph()

    pipeline_cases = [
        ("Routage hors-domaine (pas de recherche/generation attendue)",
         "Quelle est la capitale de la Tunisie ?",
         lambda s: s.get("route") == "hors_domaine"),
        ("Question factuelle complete (recherche + generation + groundedness)",
         "Qu'est-ce que le tabagisme passif ?",
         lambda s: bool((s.get("final_answer") or "").strip())),
    ]

    for name, question, is_expected in pipeline_cases:
        def run_pipeline_case():
            state = app.invoke({
                "question": question,
                "retry_count": 0,
                "rewrite_history": [],
                "groundedness_fix_count": 0,
            })
            detail = f"route={state.get('route')} reponse={'oui' if state.get('final_answer') else 'non'}"
            if not is_expected(state):
                raise RuntimeError(f"resultat inattendu -- {detail}")
            return detail
        run_check(name, run_pipeline_case)

    print_results()

    failed = [n for n, ok, _, _ in results if not ok]
    if failed:
        print(f"ECHEC : {len(failed)} verification(s) en echec -- voir le detail ci-dessus.")
        sys.exit(1)
    print("Tout est OK -- pipeline operationnel.")
    sys.exit(0)


if __name__ == "__main__":
    main()
