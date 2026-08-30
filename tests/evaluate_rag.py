import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

from src.rag.chain import get_rag_chain
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

DATASET_PATH = Path("tests/test_dataset.json")

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def evaluate():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Fichier de test introuvable : {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    rag_chain = get_rag_chain()
    embeddings_model = MistralAIEmbeddings(model="mistral-embed")

    results = []
    print(f"--- Démarrage de l'évaluation sur {len(test_cases)} cas de test ---\n")

    for idx, case in enumerate(test_cases, start=1):
        question = case["question"]
        ground_truth = case["ground_truth"]

        # Génération RAG
        generated_answer = rag_chain.invoke(question)

        # Calcul de similarité sémantique via embeddings
        vec_gen = embeddings_model.embed_query(generated_answer)
        vec_gt = embeddings_model.embed_query(ground_truth)
        similarity = cosine_similarity(vec_gen, vec_gt)

        # Classification basique
        status = "Correct" if similarity >= 0.75 else ("Partiel" if similarity >= 0.5 else "Incorrect")

        result_item = {
            "id": idx,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "similarity_score": round(similarity, 4),
            "status": status
        }
        results.append(result_item)

        print(f"Test #{idx} : {question}")
        print(f"Score : {result_item['similarity_score']} | Statut : {status}")
        print(f"Réponse RAG : {generated_answer[:120]}...\n")

    avg_similarity = sum(r["similarity_score"] for r in results) / len(results)
    print("=" * 40)
    print(f"Score moyen de similarité sémantique : {avg_similarity:.4f}")
    print("=" * 40)

    # Sauvegarde des métriques
    output_report = Path("tests/evaluation_report.json")
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump({"average_similarity": avg_similarity, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"Rapport sauvegardé dans {output_report}")

if __name__ == "__main__":
    evaluate()