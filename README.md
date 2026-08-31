# Puls-Events - Assistant Intelligent de Recommandation Culturelle (POC RAG)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-4B8BBE?style=for-the-badge)
![Mistral AI](https://img.shields.io/badge/Mistral%20AI-FA520F?style=for-the-badge&logo=mistralai&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge)

Ce projet implémente un système RAG (*Retrieval-Augmented Generation*) complet pour recommander des événements culturels récents à partir des données ouvertes d'Open Agenda.

---

## 1. Architecture Technique

- **Source de Données** : API Open Agenda (filtrage géographique Nouvelle-Aquitaine / Bordeaux, historique < 1 an et à venir).
- **Ingestion & Découpage** : `RecursiveCharacterTextSplitter` (chunks de 500 caractères, overlap de 50).
- **Base Vectorielle** : Index Faiss (`IndexFlatL2` via `faiss-cpu`) pour la recherche par similarité cosinus / distance vectorielle.
- **Modèles Utilisés** :
  - Embeddings : `mistral-embed` (Mistral AI).
  - Génération : `mistral-small-latest` (Mistral AI) avec contrôle strict des hallucinations via prompt engineering.
- **Orchestration** : LangChain (LangChain Expression Language - LCEL).
- **Exposition Web** : API REST FastAPI asynchrone avec validation de schéma Pydantic.
- **Conteneurisation & Orchestration** : Dockerfile optimisé avec `uv` et `docker-compose.yml` rattaché à la stack `openclassrooms`.

---

## 2. Structure du Projet

```text
6_design_and_deploy_rag_system/
├── data/
│   ├── raw/                   # Données brutes au format Parquet
│   └── faiss_index/           # Index vectoriel Faiss persisté (.faiss, .pkl)
├── src/
│   ├── api/                   # Point d'entrée FastAPI (main.py)
│   ├── data/                  # Script d'ingestion Open Agenda (fetch_openagenda.py)
│   ├── indexing/               # Script d'indexation vectorielle (build_index.py)
│   └── rag/                   # Chaîne LangChain LCEL (chain.py)
├── tests/
│   ├── api_test.py             # Tests fonctionnels des endpoints HTTP
│   ├── evaluate_rag.py         # Script d'évaluation automatisée
│   ├── test_dataset.json       # Jeu de test annoté (Vérités terrain)
│   └── evaluation_report.json  # Métriques de performance RAG
├── Dockerfile                  # Configuration de l'image Docker applicative
├── docker-compose.yml           # Déploiement multi-services (Stack openclassrooms)
├── pyproject.toml               # Dépendances du projet
├── uv.lock                      # Verrouillage déterministe des versions
└── README.md                    # Documentation technique
```

---

## 3. Installation et Reproduction

### Prérequis

- Python 3.11+
- Gestionnaire de paquets `uv`
- Docker & Docker Compose
- Clé d'API Mistral AI

### Configuration

Cloner le dépôt et initialiser l'environnement virtuel avec les dépendances verrouillées :

```bash
uv sync --frozen
```

Créer un fichier `.env` à la racine du projet :

```env
OPENAGENDA_API_KEY=""
MISTRAL_API_KEY="votre_cle_api_mistral"
```

### Pipeline de Données & Indexation

Extraction des événements Open Agenda :

```bash
uv run python src/data/fetch_openagenda.py
```

Génération des embeddings et de l'index Faiss :

```bash
uv run python src/indexing/build_index.py
```

---

## 4. Lancement de l'Application

### Option A : Déploiement via Docker Compose (Recommandé)

Le fichier `docker-compose.yml` monte le service au sein de la stack globale `openclassrooms` :

```bash
# Construction et lancement du conteneur en arrière-plan
docker compose up -d --build

# Vérification du statut du conteneur
docker compose ps

# Consultation des logs de l'API
docker compose logs -f puls_events_rag
```

### Option B : Lancement Manuel en Local

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

- **Swagger UI** (Documentation interactive) : http://localhost:8000/docs
- **Vérification de santé** (Healthcheck) : http://localhost:8000/health

---

## 5. Endpoints de l'API

| Méthode | Route      | Description                          | Payload d'exemple                                      |
|---------|------------|---------------------------------------|----------------------------------------------------------|
| GET     | `/health`  | Statut opérationnel de l'API          | Aucun                                                     |
| POST    | `/ask`     | Interrogation du système RAG          | `{"question": "Quels sont les concerts prévus ?"}`       |
| POST    | `/rebuild` | Reconstruction à chaud de l'index Faiss | Aucun                                                   |

---

## 6. Tests et Évaluation de la Qualité

### Tests Fonctionnels de l'API

```bash
uv run python -m pytest tests/api_test.py
```

### Pipeline d'Évaluation Sémantique RAG

```bash
uv run python tests/evaluate_rag.py
```

### Métriques Observées

- **Similarité Sémantique Moyenne** : 0.8851 sur le jeu de test annoté.
- **Garde-fous** : Absence totale d'hallucination constatée lors des requêtes hors catalogue (réponse négative explicite).

---

## 7. Arrêt du Service

Pour arrêter et nettoyer les conteneurs Docker sans supprimer les volumes de données :

```bash
docker compose down
```

---

## 8. Pistes d'Amélioration
- **Hybridation de la recherche** : Combiner Faiss avec une recherche lexicale (BM25) pour améliorer le repérage de noms propres ou artistes spécifiques.
- **Re-ranking** : Intégrer un modèle de reranking (ex: Cohere Rerank ou Mistral Rerank) avant la génération pour optimiser l'ordre des contextes.
- **Historique de conversation** : Ajouter une mémoire de session Redis pour gérer le contexte multi-tours.