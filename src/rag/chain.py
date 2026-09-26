"""
Chaîne RAG (Retrieval-Augmented Generation) pour Puls-Events.

Orchestre, via LangChain (LCEL) :
1. La recherche des documents pertinents dans l'index Faiss.
2. La génération d'une réponse en langage naturel par un LLM gratuit
   (NVIDIA NIM ou Hugging Face local), à partir du contexte récupéré.
"""
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.rag.embeddings_provider import get_embeddings
from src.rag.llm_provider import get_llm

load_dotenv()

INDEX_DIR = Path("data/faiss_index")

PROMPT_TEMPLATE = """Tu es un assistant expert pour la plateforme Puls-Events, spécialisé dans la recommandation d'événements culturels.
Réponds à la question de l'utilisateur de manière précise, chaleureuse et structurée, en te basant EXCLUSIVEMENT sur le contexte fourni ci-dessous.
Si les informations fournies ne permettent pas de répondre, indique clairement que tu ne trouves pas d'événement correspondant dans le catalogue actuel.

Contexte :
{context}

Question :
{question}

Réponse :"""


def format_docs(docs) -> str:
    """Concatène le contenu des documents récupérés en un seul bloc de contexte."""
    if not docs:
        return "Aucun événement trouvé dans le catalogue."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def load_vector_store(index_dir: Path = INDEX_DIR) -> FAISS:
    """Charge l'index Faiss persisté sur disque."""
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index introuvable : {index_dir}. "
            "Exécutez d'abord src/indexing/build_index.py"
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )


def get_rag_chain(index_dir: Path = INDEX_DIR, k: int = 3, temperature: float = 0.2):
    """Construit la chaîne RAG complète (retriever + prompt + LLM + parseur)."""
    vector_store = load_vector_store(index_dir)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = get_llm(temperature=temperature)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    rag_chain = get_rag_chain()
    query = "Quels sont les événements musicaux ou expositions prévus ?"
    print(f"Question test : {query}\n")
    response = rag_chain.invoke(query)
    print("Réponse du système RAG :")
    print(response)
