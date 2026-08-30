from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Index introuvable : {INDEX_DIR}. Exécutez d'abord src/indexing/build_index.py")

    # Chargement de l'index et des embeddings
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Chaîne LCEL (LangChain Expression Language)
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