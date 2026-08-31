from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.chain import get_rag_chain
from src.indexing.build_index import build_vector_store

# Initialisation globale de la chaîne RAG
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    try:
        rag_chain = get_rag_chain()
        print("Chaîne RAG initialisée avec succès.")
    except Exception as e:
        print(f"Index non prêt au démarrage : {e}")
    yield

app = FastAPI(
    title="Puls-Events Cultural Assistant API",
    description="API REST exposant un système RAG pour la recommandation d'événements culturels récents.",
    version="1.0.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    question: str = Field(
        ..., 
        json_schema_extra={"example": "Quels sont les concerts prévus ce mois-ci ?"}
    )

class QueryResponse(BaseModel):
    question: str
    response: str

class StatusResponse(BaseModel):
    status: str
    message: str

@app.get("/health", response_model=StatusResponse, tags=["Monitoring"])
def health_check():
    return StatusResponse(status="ok", message="API Puls-Events opérationnelle.")

@app.post("/ask", response_model=QueryResponse, tags=["RAG"])
def ask_question(request: QueryRequest):
    global rag_chain
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")
    
    if rag_chain is None:
        try:
            rag_chain = get_rag_chain()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Index vectoriel non disponible : {e}")
            
    try:
        answer = rag_chain.invoke(request.question)
        return QueryResponse(question=request.question, response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {str(e)}")

@app.post("/rebuild", response_model=StatusResponse, tags=["Indexation"])
def rebuild_index():
    global rag_chain
    try:
        build_vector_store()
        rag_chain = get_rag_chain()
        return StatusResponse(status="success", message="Index Faiss reconstruit et rechargé avec succès.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec de la reconstruction de l'index : {str(e)}")