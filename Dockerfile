FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Installation de uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copie des fichiers de configuration
COPY pyproject.toml uv.lock ./

# Installation des dépendances dans le conteneur
RUN uv sync --frozen --no-cache

# Copie du code source et des données vectorisées
COPY src/ ./src/
COPY data/ ./data/

# Exposition du port
EXPOSE 8000

# Commande d'exécution
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]