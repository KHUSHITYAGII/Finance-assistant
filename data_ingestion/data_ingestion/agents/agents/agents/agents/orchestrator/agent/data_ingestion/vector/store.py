"""
Vector database implementation for document storage and retrieval
"""

import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import pickle
from datetime import datetime
from loguru import logger
import hashlib

from config.settings import settings


class VectorStore:
    """Vector database for storing and retrieving document embeddings"""
    
    def __init__(self, embedding_dimension: int = 1536):
        """
        Initialize the vector store
        
        Args:
            embedding_dimension: Dimension of the embeddings
        """
        self.embedding_dimension = embedding_dimension
        self.index = None
        self.documents = []
        self.embedding_model = settings.EMBEDDING_MODEL
        
        # Create empty index
        self._create_empty_index()
        
        # Directory for saving indexes
        os.makedirs("data/indexes", exist_ok=True)
    
    def _create_empty_index(self):
        """Create an empty FAISS index"""
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
    
    async def add_texts(self, 
                       texts: List[str], 
                       metadatas: Optional[List[Dict[str, Any]]] = None,
                       embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """
        Add texts to the vector store
        
        Args:
            texts: List of text chunks to add
            metadatas: List of metadata dicts for each text
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = await self._get_embeddings(texts)
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Generate document IDs
        doc_ids = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            doc_id = self._generate_id(text)
            doc_ids.append(doc_id)
            
            # Store document with metadata
            self.documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "embedding_index": len(self.documents)
            })
        
        return doc_ids
    
    async def similarity_search(self, 
                              query: str, 
                              k: int = 5,
                              threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform similarity search against the vector store
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of documents with similarity scores
        """
        if not self.documents:
            return []
        
        # Get query embedding
        query_embedding = await self._get_embeddings([query])
        query_array = np.array(query_embedding).astype('float32')
        
        # Search index
        num_docs = min(k, len(self.documents))
        if num_docs == 0:
            return []
            
        distances, indices = self.index.search(query_array, num_docs)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            # Convert distance to similarity score (1 - normalized distance)
            similarity = 1.0 - min(dist / 10.0, 1.0)  # Simple normalization
            
            if similarity < threshold:
                continue
                
            # Find the document
            for doc in self.documents:
                if doc["embedding_index"] == idx:
                    results.append({
                        "id": doc["id"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "similarity": similarity
                    })
                    break
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results
    
    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts
        
        In a real implementation, this would call an embedding API
        Here we'll simulate it for demonstration purposes
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        # For demo purposes, we'll generate deterministic "fake" embeddings
        # based on the content hash
        
        def generate_fake_embedding(text: str) -> List[float]:
            """Generate a fake embedding based on text hash"""
            # Create a deterministic hash of the text
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Use the hash to seed numpy's random generator
            seed = int.from_bytes(hash_bytes[:4], byteorder='little')
            rng = np.random.RandomState(seed)
            
            # Generate a fake embedding vector
            embedding = rng.randn(self.embedding_dimension).astype('float32')
            
            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding.tolist()
        
        return [generate_fake_embedding(text) for text in texts]
    
    def _generate_id(self, text: str) -> str:
        """Generate a document ID based on content hash"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def save_index(self, file_path: str = "data/indexes/faiss_index.bin"):
        """
        Save the FAISS index to disk
        
        Args:
            file_path: Path to save the index
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, file_path)
            
            # Save documents
            docs_path = file_path.replace(".bin", "_docs.pkl")
            with open(docs_path, "wb") as f:
                pickle.dump(self.documents, f)
                
            logger.info(f"Saved index with {len(self.documents)} documents to {file_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def load_index(self, file_path: str = "data/indexes/faiss_index.bin") -> bool:
        """
        Load the FAISS index from disk
        
        Args:
            file_path: Path to the index file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Index file not found: {file_path}")
                return False
                
            # Load FAISS index
            self.index = faiss.read_index(file_path)
            
            # Load documents
            docs_path = file_path.replace(".bin", "_docs.pkl")
            if os.path.exists(docs_path):
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                    
            logger.info(f"Loaded index with {len(self.documents)} documents from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False