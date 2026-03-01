
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
import logging
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Import optimized backend
from pharma_rag import IngestionEngine, ChunkingEngine, VectorDatabase, RAGController

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Global State
class State:
    vector_db: Optional[VectorDatabase] = None
    rag_controller: Optional[RAGController] = None

state = State()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Backend...")
    try:
        if os.path.exists("./chroma_db"):
            state.vector_db = VectorDatabase()
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                state.rag_controller = RAGController(state.vector_db, groq_key)
            logger.info("Knowledge Base Loaded.")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
    
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(title="Pharma RAG API", lifespan=lifespan)

# Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    images: List[str]

# -----------------------------------------------------------------------------
# Background Task: File Processing
# -----------------------------------------------------------------------------
def process_file_task(file_path: str, file_name: str):
    logger.info(f"Starting background processing for {file_name}")
    try:
        # 1. Ingestion
        ingestion = IngestionEngine()
        doc_object, _, image_map = ingestion.process_file(file_path)

        # 2. Chunking
        chunking = ChunkingEngine()
        chunks = chunking.chunk_document(doc_object, image_map)

        # 3. Store
        if not state.vector_db:
            logger.info("Initializing global VectorDatabase to prevent OOM...")
            state.vector_db = VectorDatabase()
            
        state.vector_db.create_or_update_vector_store(chunks)
        
        # Update global state
        if not state.rag_controller:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                state.rag_controller = RAGController(state.vector_db, groq_key)
            
        logger.info(f"Completed processing {file_name}")
    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "components": {"rag": state.rag_controller is not None}}

@app.get("/files")
def list_files():
    if state.vector_db:
        return {"files": state.vector_db.list_ingested_files()}
    return {"files": []}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_dir = "data/temp"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.filename)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Process synchronously so we can report success immediately
    process_file_task(file_path, file.filename)
    
    return {"message": f"File '{file.filename}' processed successfully.", "status": "completed"}

@app.delete("/files/{filename}")
def delete_file(filename: str):
    if not state.vector_db:
         raise HTTPException(status_code=404, detail="Database not initialized")
    
    success = state.vector_db.delete_file(filename)
    if success:
        return {"message": f"Deleted {filename}"}
    else:
        raise HTTPException(status_code=500, detail="Deletion failed")

@app.post("/query")
def query_rag(request: QueryRequest):
    if not state.rag_controller:
        raise HTTPException(status_code=503, detail="RAG System not ready. Please upload a document.")
    
    try:
        # We use the synchronous query for the API (Streaming is harder via simple REST, usually use SSE)
        # For this version, we'll return full response. 
        # To support streaming, we'd need a StreamingResponse.
        
        # Let's use the standard query method which wraps everything
        result = state.rag_controller.query(request.question)
        
        # Extract sources and images
        context_docs = result["context"]
        sources = list(set([doc.metadata.get("source", "unknown") for doc in context_docs]))
        
        images = set()
        for doc in context_docs:
            if "images" in doc.metadata:
                imgs = doc.metadata["images"].split(",")
                for img in imgs:
                    if img.strip():
                        # Return relative path for frontend to fetch
                        # API could serve them, but for now we assume shared FS or frontend can access
                        images.add(img.strip())
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "images": list(images)
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
