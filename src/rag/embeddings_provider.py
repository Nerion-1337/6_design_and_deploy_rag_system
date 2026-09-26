"""
Fournisseur d'embeddings pour le système RAG Puls-Events.

Utilise un modèle Hugging Face local (sentence-transformers) exécuté sur CPU.
Ce choix garantit une solution 100% gratuite et reproductible, sans dépendance
à une clé API externe pour la vectorisation (contrairement à l'API Mistral
initialement utilisée, qui n'est plus gratuite).
"""
import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Instancie (une seule fois, via cache) le modèle d'embeddings local.

    Le modèle par défaut (all-MiniLM-L6-v2) est léger (~80 Mo), tourne sur CPU
    et ne nécessite aucune clé API : il est donc utilisable gratuitement et
    sans dépendance à un service tiers payant.
    """
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
