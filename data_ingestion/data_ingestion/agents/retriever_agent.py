import os
import numpy as np
import faiss
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import pickle
import json
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Finance Retriever Agent")

# Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Path to store vector database
INDEX_DIR = os.path.join(os.path.dirname(__file__), "../data/vector_store")
os.makedirs(INDEX_DIR, exist_ok=True)

# Path to store documents
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "../data/documents.pkl")

class RetrievalRequest(BaseModel):
    query: str
    market_data: Optional[Dict[str, Any]] = None
    news_data: Optional[Dict[str, Any]] = None

class IndexDocumentRequest(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

def load_or_create_index():
    """Load existing index or create a new one if not exists."""
    index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
    documents_path = DOCUMENTS_PATH
    
    if os.path.exists(index_path) and os.path.exists(documents_path):
        # Load existing index and documents
        index = faiss.read_index(index_path)
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)
        logger.info(f"Loaded existing index with {index.ntotal} vectors")
        return index, documents
    else:
        # Create new index and empty documents list
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        documents = []
        
        # Save empty index and documents
        faiss.write_index(index, index_path)
        with open(documents_path, 'wb') as f:
            pickle.dump(documents, f)
        
        logger.info(f"Created new index with dimension {dimension}")
        return index, documents

# Load or create index on startup
index, documents = load_or_create_index()

# Seed some initial financial data for demonstration
def seed_initial_data():
    global index, documents
    
    if len(documents) == 0:
        # Add some seed financial data
        seed_documents = [
            {
                "title": "TSMC Earnings Report Q1 2023",
                "content": "TSMC reported earnings of $4.93 per share, beating estimates by 4%. Revenue rose to $17.6 billion, up 14% year-over-year.",
                "metadata": {
                    "ticker": "2330.TW",
                    "company": "TSMC",
                    "date": "2023-04-20",
                    "type": "earnings"
                }
            },
            {
                "title": "Samsung Electronics Q1 2023 Results",
                "content": "Samsung Electronics reported earnings of $0.95 per share, missing estimates by 2%. Revenue was $54.05 billion, down 4% year-over-year.",
                "metadata": {
                    "ticker": "005930.KS",
                    "company": "Samsung",
                    "date": "2023-04-27",
                    "type": "earnings"
                }
            },
            {
                "title": "Asia Tech Stocks Market Analysis",
                "content": "Asia tech stocks have shown resilience despite global economic challenges. The sector currently represents about 18% of total AUM in diversified portfolios.",
                "metadata": {
                    "region": "asia",
                    "sector": "tech",
                    "date": "2023-05-15",
                    "type": "analysis"
                }
            },
            {
                "title": "Rising Bond Yields Impact on Tech Stocks",
                "content": "Rising bond yields have led to cautious sentiment in Asia's tech sector. Investors are monitoring the impact on future growth prospects.",
                "metadata": {
                    "region": "global",
                    "sector": "tech",
                    "date": "2023-05-10",
                    "type": "analysis"
                }
            }
        ]
        
        # Index the seed documents
        for doc in seed_documents:
            index_document(doc)
        
        logger.info(f"Seeded {len(seed_documents)} documents into the index")

def index_document(document):
    """Index a document into the vector database."""
    global index, documents
    
    # Create the text to embed (title + content)
    text_to_embed = f"{document['title']}: {document['content']}"
    
    # Generate embedding
    embedding = model.encode([text_to_embed])[0]
    embedding = np.array([embedding]).astype('float32')
    
    # Add to FAISS index
    index.add(embedding)
    
    # Store document
    documents.append(document)
    
    # Save updated index and documents
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_index.bin"))
    with open(DOCUMENTS_PATH, 'wb') as f:
        pickle.dump(documents, f)
    
    return len(documents) - 1  # Return the document ID

@app.post("/index")
async def add_document(request: IndexDocumentRequest):
    """Add a document to the vector database."""
    try:
        document = {
            "title": request.title,
            "content": request.content,
            "metadata": request.metadata or {}
        }
        
        doc_id = index_document(document)
        
        return {
            "success": True,
            "data": {
                "document_id": doc_id
            }
        }
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

@app.post("/retrieve")
async def retrieve(request: RetrievalRequest):
    """Retrieve relevant documents based on the query."""
    try:
        # Generate embedding for the query
        query_embedding = model.encode([request.query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Perform search
        k = 5  # Number of results to retrieve
        distances, indices = index.search(query_embedding, k)
        
        # Get retrieved documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(documents) and idx >= 0:
                doc = documents[idx]
                results.append({
                    "document": doc,
                    "distance": float(distances[0][i]),
                    "score": 1.0 - min(float(distances[0][i]) / 10.0, 0.99)  # Convert distance to similarity score
                })
        
        # Calculate confidence based on scores
        if results:
            avg_score = sum(r["score"] for r in results) / len(results)
            confidence = avg_score
        else:
            confidence = 0.0
        
        # Enhance results with current market data
        enhanced_results = []
        if request.market_data:
            for result in results:
                # Add market context if relevant
                if "ticker" in result["document"].get("metadata", {}):
                    ticker = result["document"]["metadata"]["ticker"]
                    # Look for this ticker in the market data
                    stocks = request.market_data.get("stocks", {})
                    if ticker in stocks:
                        result["market_context"] = stocks[ticker]
                
                enhanced_results.append(result)
        else:
            enhanced_results = results
        
        return {
            "success": True,
            "data": {
                "results": enhanced_results,
                "query": request.query
            },
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error in retrieve: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

# Seed initial data on startup
seed_initial_data()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)