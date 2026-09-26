"""
Évaluation automatisée de la qualité du système RAG.

Deux niveaux de métriques :
1. Similarité sémantique cosinus (toujours calculée, embeddings locaux
   gratuits — plus de dépendance à l'API Mistral).
2. Métriques Ragas (faithfulness, answer_relevancy, context_precision) si le
   package `ragas` est installé et qu'un LLM est configuré ; sinon, cette
   étape est ignorée proprement (utile en CI sans clé API).
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.rag.chain import get_rag_chain, load_vector_store
from src.rag.embeddings_provider import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path("tests/test_dataset.json")
REPORT_PATH = Path("tests/evaluation_report.json")


def cosine_similarity(v1, v2) -> float:
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))


def classify(similarity: float) -> str:
    if similarity >= 0.75:
        return "Correct"
    if similarity >= 0.5:
        return "Partiel"
    return "Incorrect"


def run_similarity_evaluation(test_cases: list[dict]) -> list[dict]:
    """Génère une réponse pour chaque question et la compare à la vérité terrain."""
    rag_chain = get_rag_chain()
    embeddings_model = get_embeddings()

    results = []
    for idx, case in enumerate(test_cases, start=1):
        question = case["question"]
        ground_truth = case["ground_truth"]

        try:
            generated_answer = rag_chain.invoke(question)
        except Exception as exc:
            logger.warning("Échec de génération pour '%s...' : %s", question[:30], exc)
            generated_answer = ""

        vec_gen = embeddings_model.embed_query(generated_answer or " ")
        vec_gt = embeddings_model.embed_query(ground_truth)
        similarity = cosine_similarity(vec_gen, vec_gt)
        status = classify(similarity)

        result_item = {
            "id": idx,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "similarity_score": round(similarity, 4),
            "status": status,
        }
        results.append(result_item)
        logger.info("Test #%d : %s -> score=%.4f (%s)", idx, question, similarity, status)

    return results


def run_ragas_evaluation(test_cases: list[dict], results: list[dict]) -> Optional[dict]:
    """
    Calcule des métriques Ragas complémentaires (fidélité, pertinence de la
    réponse, précision du contexte) si le package est disponible et qu'un LLM
    est configuré. Retourne None sinon (dégradation silencieuse, utile en CI).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        from src.rag.llm_provider import get_llm
    except ImportError:
        logger.info("Ragas non installé : métriques avancées ignorées.")
        return None

    try:
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        dataset_rows = []
        for case, result in zip(test_cases, results):
            contexts = [doc.page_content for doc in retriever.invoke(case["question"])]
            dataset_rows.append(
                {
                    "question": case["question"],
                    "answer": result["generated_answer"],
                    "contexts": contexts,
                    "ground_truth": case["ground_truth"],
                }
            )

        dataset = Dataset.from_list(dataset_rows)
        llm = get_llm()
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
        )
        return dict(ragas_result)
    except Exception as exc:
        logger.warning("Évaluation Ragas ignorée (erreur : %s)", exc)
        return None


def evaluate() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Fichier de test introuvable : {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    logger.info("--- Démarrage de l'évaluation sur %d cas de test ---", len(test_cases))
    results = run_similarity_evaluation(test_cases)

    avg_similarity = sum(r["similarity_score"] for r in results) / len(results)
    logger.info("Score moyen de similarité sémantique : %.4f", avg_similarity)

    ragas_metrics = run_ragas_evaluation(test_cases, results)

    report = {
        "average_similarity": avg_similarity,
        "ragas_metrics": ragas_metrics,
        "details": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Rapport sauvegardé dans %s", REPORT_PATH)
    return report


if __name__ == "__main__":
    evaluate()
