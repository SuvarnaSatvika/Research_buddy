import os
import shutil
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from ingest import build_vector_database
from cite import cite_answers
from multi_search import multi_search
from analyze import timeline_tracker, compare_multiple_papers

app = FastAPI(title="Research Buddy API")
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query : str
    paper_title :Optional[str] = "ALL"

# --- Pydantic Data Models ---
class CompareRequest(BaseModel):
    topic: str
    paper_titles: List[str]


# --- Compare Endpoint ---
@app.post("/api/analyze/compare")
async def compare_documents(request: CompareRequest):
    """
    Side-by-side contrastive analysis across any number of selected papers.
    """
    # Remove duplicates and filter out placeholder 'ALL'
    valid_titles = list(dict.fromkeys([p for p in request.paper_titles if p != "ALL"]))

    if len(valid_titles) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please select at least 2 distinct papers from the dropdowns to run a comparison."
        )

    try:
        response_text = compare_multiple_papers(request.topic, valid_titles)
        return {"status": "success", "data": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    ingested_files = []
    errors = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            errors.append(f"'{file.filename}' skipped (only PDFs are supported).")
            continue

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        try:
            # Save the PDF locally
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Ingest into ChromaDB
            build_vector_database(file_path)
            ingested_files.append(file.filename)

        except Exception as e:
            errors.append(f"Failed to ingest '{file.filename}': {str(e)}")

    if not ingested_files and errors:
        raise HTTPException(status_code=500, detail=" | ".join(errors))

    return {
        "status": "success",
        "message": f"Successfully processed {len(ingested_files)} file(s).",
        "ingested_files": ingested_files,
        "warnings": errors
    }

@app.get("/api/papers")
async def get_all_papers():
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(
            collection_name="research_papers",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Fetch metadata from ChromaDB
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        # Extract unique titles
        titles = list(set([m.get("title") for m in metadatas if m and "title" in m]))
        return {"status": "success", "papers": titles}
    except Exception as e:
        return {"status": "error", "papers": [], "detail": str(e)}
    
@app.post("/api/ask")
async def ask_copilot(request: QueryRequest):
    try:
        response_text = cite_answers(request.query, request.paper_title)

        return {"status": "success", "data": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/timeline")
async def generate_timeline(request: QueryRequest):
    try:
        response_text = timeline_tracker(request.query)
        return {"status": "success", "data": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/compare")
async def compare_documents(request: CompareRequest):
    try:
        response_text = compare_papers(request.topic, request.paper1_title, request.paper2_title)
        return {"status": "success", "data": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research/deep-dive")
async def deep_dive(request: QueryRequest):
    """
    Multi-hop reasoning endpoint for complex research queries.
    """
    try:
        response_text = multi_search(request.query)
        return {"status": "success", "data": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "active", "message": "Research Copilot API is online"}
