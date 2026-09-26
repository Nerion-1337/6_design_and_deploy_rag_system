# Puls-Events - Assistant Intelligent de Recommandation Culturelle (POC RAG)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-4B8BBE?style=for-the-badge)
![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-Free%20API-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Local%20Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge)

Ce projet implémente un système RAG (*Retrieval-Augmented Generation*) complet pour recommander des événements culturels récents à partir des données ouvertes d'Open Agenda.

> **⚠️ Changement important** : l'API Mistral n'étant plus gratuite, le système utilise désormais des embeddings **Hugging Face locaux** (gratuits, sans clé API) et un LLM au choix entre **NVIDIA NIM** (API gratuite avec clé) ou **Hugging Face local** (aucune clé requise). Voir la section [Fournisseurs de modèles](#5-fournisseurs-de-modèles-gratuits).

---

## 1. Architecture Technique

![Schéma d'architecture](docs/architecture.png)

> Explication détaillée de chaque composant : voir [`docs/architecture.md`](docs/architecture.md) (fichier source vectoriel réutilisable : [`docs/architecture.svg`](docs/architecture.svg)).

- **Source de Données** : API Open Agenda (filtrage géographique Nouvelle-Aquitaine / Bordeaux, historique < 1 an et à venir), avec pagination automatique, retries/backoff et nettoyage/enrichissement des données.
- **Ingestion & Découpage** : `RecursiveCharacterTextSplitter` (chunks de 500 caractères, overlap de 50).
- **Base Vectorielle** : Index Faiss (`IndexFlatL2` via `faiss-cpu`) pour la recherche par similarité.
- **Modèles Utilisés (100% gratuits)** :
  - Embeddings : `sentence-transformers/all-MiniLM-L6-v2` exécuté **localement** (aucune clé API).
  - Génération : au choix via la variable `RAG_LLM_PROVIDER` —
    - `nvidia` : `meta/llama-3.2-11b-vision-instruct` via l'[API NVIDIA NIM](https://build.nvidia.com) (gratuite, clé personnelle requise ; vérifiez sur le catalogue que le modèle choisi porte bien le tag **"Free Endpoint"**, sinon l'appel échoue) ;
    - `huggingface_local` : `Qwen/Qwen2.5-1.5B-Instruct` exécuté localement (aucune clé, hors ligne).
- **Orchestration** : LangChain (LangChain Expression Language - LCEL).
- **Exposition Web** : API REST FastAPI asynchrone avec validation de schéma Pydantic.
- **Conteneurisation & Orchestration** : Dockerfile optimisé avec `uv` et `docker-compose.yml` rattaché à la stack `openclassrooms`.

---

## 2. Structure du Projet

```text
puls-events-rag/
├── .github/workflows/
│   └── tests.yml                            # CI : tests + évaluation automatisée
├── docs/
│   ├── architecture.svg                      # Schéma d'architecture (vectoriel)
│   ├── architecture.png                      # Schéma d'architecture (image, pour rapport/PPTX)
│   └── architecture.md                       # Explication détaillée de chaque composant
├── data/
│   ├── raw/                                  # Données brutes (Parquet)
│   └── faiss_index/                          # Index vectoriel Faiss persisté
├── src/
│   ├── api/main.py                           # API FastAPI (/health, /metadata, /ask, /rebuild)
│   ├── data/fetch_openagenda.py              # Ingestion robuste (pagination, retries, nettoyage, enrichissement)
│   ├── indexing/build_index.py               # Chunking + vectorisation + index Faiss
│   └── rag/
│       ├── chain.py                          # Chaîne LangChain LCEL (retriever + prompt + LLM)
│       ├── embeddings_provider.py            # Fournisseur d'embeddings gratuits (local)
│       └── llm_provider.py                   # Fournisseur LLM gratuit (NVIDIA NIM / HF local)
├── tests/
│   ├── test_fetch_openagenda.py              # Tests : récupération, pagination, nettoyage, enrichissement
│   ├── test_build_index.py                   # Tests : chunking, vectorisation, persistance de l'index
│   ├── test_chain.py                         # Tests : interrogation du retriever, génération
│   ├── test_providers.py                     # Tests : sélection des fournisseurs LLM/embeddings
│   ├── test_evaluate_rag.py                  # Tests : script d'évaluation (similarité, classification)
│   ├── api_test.py                           # Tests fonctionnels des endpoints HTTP
│   ├── evaluate_rag.py                       # Évaluation automatisée (similarité + Ragas optionnel)
│   └── test_dataset.json                     # Jeu de test annoté (vérités terrain)
├── scripts/
│   └── run_all_tests.sh                      # Lance tests + évaluation en une commande
├── Dockerfile                                 # Image applicative (API + suite de tests)
├── docker-compose.yml                         # Déploiement (stack openclassrooms)
├── Makefile                                   # Commandes usuelles (install, test, evaluate, docker...)
├── pyproject.toml                             # Dépendances du projet
├── uv.lock                                    # Verrouillage déterministe des versions
└── README.md                                  # Documentation technique
```

---

## 3. Installation et Reproduction

### Prérequis

- Python 3.11+
- Gestionnaire de paquets `uv`
- Docker & Docker Compose
- (Optionnel) Une clé API gratuite NVIDIA NIM si vous utilisez ce fournisseur

### Configuration

```bash
uv sync --frozen --extra dev
cp .env.example .env
```

Renseignez `.env` selon le fournisseur choisi (voir [section 5](#5-fournisseurs-de-modèles-gratuits)).

### Pipeline de Données & Indexation

```bash
uv run python -m src.data.fetch_openagenda
uv run python -m src.indexing.build_index
```

---

## 4. Lancement de l'Application

### Option A : Docker Compose (recommandé)

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f puls_events_rag
```

### Option B : Lancement manuel en local

```bash
uv run python -m uvicorn src.api.main:app --reload --port 8001
```

- **Swagger UI** : http://localhost:8001/docs
- **Healthcheck** : http://localhost:8001/health

---

## 5. Fournisseurs de modèles (gratuits)

| Composant   | Fournisseur                  | Clé API requise | Exécution |
|-------------|-------------------------------|------------------|-----------|
| Embeddings  | `sentence-transformers` (HF)  | Non              | Locale (CPU) |
| LLM         | NVIDIA NIM (`RAG_LLM_PROVIDER=nvidia`) | Oui (gratuite sur build.nvidia.com) | Distante |
| LLM         | Hugging Face local (`RAG_LLM_PROVIDER=huggingface_local`) | Non | Locale |

Basculer de fournisseur ne nécessite aucune modification de code : il suffit de modifier `RAG_LLM_PROVIDER` dans `.env`.

---

## 6. Endpoints de l'API

| Méthode | Route      | Description                              | Payload d'exemple                                   |
|---------|------------|-------------------------------------------|------------------------------------------------------|
| GET     | `/health`  | Statut opérationnel de l'API              | Aucun                                                 |
| GET     | `/metadata`| État de l'index (nb de vecteurs, fournisseur LLM actif) | Aucun                                    |
| POST    | `/ask`     | Interrogation du système RAG              | `{"question": "Quels sont les concerts prévus ?"}`   |
| POST    | `/rebuild` | Reconstruction à chaud de l'index Faiss   | Aucun                                                 |

---

## 7. Tests et Évaluation de la Qualité

### Lancer toute la suite en une commande

```bash
make test-all
# ou directement :
bash scripts/run_all_tests.sh
```

Cette commande est utilisée aussi bien en local, en CI (GitHub Actions, voir `.github/workflows/tests.yml`) que dans le conteneur Docker :

```bash
docker compose run --rm puls_events_rag bash scripts/run_all_tests.sh
```

### Détail des tests unitaires et fonctionnels

| Fichier                       | Couvre |
|--------------------------------|--------|
| `tests/test_fetch_openagenda.py` | Récupération (pagination, retries, encodage), nettoyage, enrichissement |
| `tests/test_build_index.py`      | Chunking, vectorisation, persistance de l'index Faiss |
| `tests/test_chain.py`            | Interrogation du retriever, génération de la réponse |
| `tests/test_providers.py`        | Sélection des fournisseurs LLM / embeddings |
| `tests/test_evaluate_rag.py`     | Calcul de similarité, classification des résultats |
| `tests/api_test.py`              | Endpoints `/health`, `/metadata`, `/ask`, `/rebuild` |

Tous ces tests sont **mockés** (aucun appel réseau réel, aucune clé API requise) afin de pouvoir tourner de façon fiable et reproductible en CI.

```bash
uv run python -m pytest tests/ -v
```

> Sous Windows, si vous rencontrez `error: uv trampoline failed to canonicalize script path`, utilisez systématiquement `uv run python -m <outil>` plutôt que `uv run <outil>` (contourne le lanceur `.exe` généré par uv, en particulier sur des chemins avec caractères accentués).

### Évaluation sémantique automatisée

```bash
uv run python -m tests.evaluate_rag
```

- Calcule une **similarité cosinus** (embeddings locaux) entre chaque réponse générée et la réponse annotée de référence.
- Calcule en complément des **métriques Ragas** (`faithfulness`, `answer_relevancy`, `context_precision`) si le package `ragas` est installé et qu'un LLM est configuré ; sinon cette étape est ignorée proprement.
- Génère `tests/evaluation_report.json`.

### Automatisation en CI (GitHub Actions)

Le workflow `.github/workflows/tests.yml` exécute automatiquement, à chaque push/PR :
1. l'installation des dépendances (`uv sync --extra dev`) ;
2. l'ensemble des tests unitaires et fonctionnels (rapport JUnit publié en artefact) ;
3. l'évaluation automatisée du RAG si un index et une clé `NVIDIA_API_KEY` (secret GitHub) sont disponibles.

### Métriques Observées (exemple)

- **Similarité Sémantique Moyenne** : 0.8851 sur le jeu de test annoté.
- **Garde-fous** : absence d'hallucination constatée lors des requêtes hors catalogue (réponse négative explicite).

---

## 8. Arrêt du Service

```bash
docker compose down
```

---

## 9. Pistes d'Amélioration

- **Hybridation de la recherche** : combiner Faiss avec une recherche lexicale (BM25) pour mieux repérer noms propres et artistes.
- **Re-ranking** : intégrer un modèle de reranking avant la génération pour optimiser l'ordre des contextes.
- **Historique de conversation** : ajouter une mémoire de session (ex. Redis) pour un contexte multi-tours.
- **Schéma d'architecture UML** : ajouter un diagramme détaillant les composants et leurs interactions dans le rapport technique.
- **Enrichissement des données** : géocodage de secours pour les événements sans coordonnées, détection de doublons approximatifs (titres similaires).

---

## 10. Vidéo

-  [`video`](https://youtu.be/tp2cCIDNM_U)
