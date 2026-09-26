FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Installation de uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copie des fichiers de configuration
COPY pyproject.toml uv.lock ./

# Installation des dépendances, y compris celles de test/évaluation (--extra dev)
# afin de pouvoir lancer toute la suite de validation depuis le conteneur.
RUN uv sync --frozen --no-cache --extra dev

# Copie du code source, des tests, des scripts et des données vectorisées
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY data/ ./data/

RUN chmod +x scripts/run_all_tests.sh

# Exposition du port de l'API
EXPOSE 8002

# Commande par défaut : démarre l'API.
# Pour lancer l'ensemble des tests + l'évaluation automatisée à la place :
#   docker compose run --rm puls_events_rag bash scripts/run_all_tests.sh
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8002"]
