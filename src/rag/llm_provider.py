"""
Fournisseur de LLM pour le système RAG Puls-Events.

L'API Mistral payante n'est plus utilisée. Deux fournisseurs gratuits sont
supportés, sélectionnables via la variable d'environnement RAG_LLM_PROVIDER :

- "nvidia"            : API NVIDIA NIM (https://build.nvidia.com), compatible
                         OpenAI, avec un quota gratuit par clé API personnelle.
- "huggingface_local" : modèle open-source exécuté localement via
                         transformers, sans aucune clé API (fonctionne hors
                         ligne ; plus lent selon la machine, pas de coût).
"""
import os

from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_NVIDIA_MODEL = "meta/llama-3.2-11b-vision-instruct"
DEFAULT_HF_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _get_nvidia_llm(temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NVIDIA_API_KEY manquante. Créez une clé gratuite sur "
            "https://build.nvidia.com puis renseignez-la dans votre fichier .env, "
            "ou basculez RAG_LLM_PROVIDER=huggingface_local pour un modèle local "
            "ne nécessitant aucune clé."
        )
    model = os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
    return ChatOpenAI(
        model=model,
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
        temperature=temperature,
    )


def _get_huggingface_local_llm(temperature: float) -> BaseChatModel:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    model_name = os.getenv("HF_LOCAL_MODEL", DEFAULT_HF_LOCAL_MODEL)
    pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512,
            "temperature": max(temperature, 0.01),
            "do_sample": temperature > 0,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=pipeline)


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """Retourne le client LLM correspondant au fournisseur configuré (gratuit)."""
    provider = os.getenv("RAG_LLM_PROVIDER", "nvidia").lower()

    if provider == "nvidia":
        return _get_nvidia_llm(temperature)
    if provider == "huggingface_local":
        return _get_huggingface_local_llm(temperature)

    raise ValueError(
        f"RAG_LLM_PROVIDER='{provider}' inconnu. Valeurs acceptées : "
        "'nvidia', 'huggingface_local'."
    )
