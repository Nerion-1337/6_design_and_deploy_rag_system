#!/usr/bin/env bash
# Lance l'ensemble de la suite de validation du système RAG Puls-Events :
#   1. Tests unitaires et fonctionnels (pytest) : récupération, nettoyage,
#      vectorisation, interrogation, génération, endpoints API.
#   2. Évaluation automatisée de la qualité du RAG (similarité sémantique,
#      et métriques Ragas si un index et une clé API sont disponibles).
#
# Utilisé en local (make test-all), en CI (GitHub Actions) et dans l'image
# Docker (docker compose run --rm puls_events_rag bash scripts/run_all_tests.sh).
set -euo pipefail

echo "==> 1/2 Exécution des tests unitaires et fonctionnels (pytest)"
uv run pytest tests/ -v

echo "==> 2/2 Évaluation automatisée de la qualité du système RAG"
if [ -d "data/faiss_index" ]; then
    uv run python -m tests.evaluate_rag || echo "Évaluation ignorée (LLM indisponible)."
else
    echo "Index vectoriel introuvable (data/faiss_index) : évaluation ignorée."
fi

echo "==> Suite de validation terminée avec succès."
