import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import os
from pathlib import Path

# Try to import optional dependencies with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logging.warning("FAISS not installed, using fallback similarity search")
    FAISS_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    logging.warning("Pinecone not installed, cannot use Pinecone vector store")
    PINECONE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("Sentence-transformers not installed, using fallback embedding method")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class RetrieverAgent:
    """
    Agent for retrieving relevant information from vector stores
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the retriever agent
        
        Args:
            config: Configuration dictionary for retriever agent
        """
        self.config = config or {}
        self.vector_store_type = self.config.get("vector_store_type", "faiss")
        self.embedding_model = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.top_k = self.config.get("top_k", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.6)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.embedding_model)
            except Exception as e:
                logging.error(f"Error loading embedding model: {e}")
                self.model = None
        else:
            self.model = None
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> Any:
        """
        Initialize the vector store based on configuration
        
        Returns:
            Vector store instance
        """
        if self.vector_store_type == "faiss" and FAISS_AVAILABLE:
            # Check if index exists
            index_path = Path(self.config.get("faiss_index_path", "data/faiss_index"))
            if os.path.exists(index_path):
                try:
                    # Load existing index
                    index = faiss.read_index(str(index_path))
                    logging.info(f"Loaded FAISS index from {index_path}")
                    return {"index": index, "texts": self._load_texts(index_path)}
                except Exception as e:
                    logging.error(f"Error loading FAISS index: {e}")
            
            # Create new index
            try:
                d = 384  # Default dimension for all-MiniLM-L6-v2
                index = faiss.IndexFlatL2(d)
                logging.info("Created new FAISS index")
                return {"index": index, "texts": []}
            except Exception as e:
                logging.error(f"Error creating FAISS index: {e}")
                return None
        
        elif self.vector_store_type == "pinecone" and PINECONE_AVAILABLE:
            # Initialize Pinecone
            api_key = self.config.get("pinecone_api_key")
            environment = self.config.get("pinecone_environment")
            index_name = self.config.get("pinecone_index_name")
            
            if api_key and environment and index_name:
                try:
                    pinecone.init(api_key=api_key, environment=environment)
                    index = pinecone.Index(index_name)
                    logging.info(f"Connected to Pinecone index {index_name}")
                    return {"index": index}
                except Exception as e:
                    logging.error(f"Error connecting to Pinecone: {e}")
                    return None
            else:
                logging.error("Missing Pinecone configuration parameters")
                return None
        
        else:
            logging.warning(f"Unsupported vector store type: {self.vector_store_type}")
            # Fallback to dictionary-based store
            return {"texts": [], "embeddings": []}
    
    def _load_texts(self, index_path: Path) -> List[str]:
        """
        Load texts associated with FAISS index
        
        Args:
            index_path: Path to FAISS index
            
        Returns:
            List of text documents
        """
        texts_path = index_path.with_suffix(".txt")
        if os.path.exists(texts_path):
            try:
                with open(texts_path, "r", encoding="utf-8") as f:
                    return f.read().split("\n===DOCUMENT_SEPARATOR===\n")
            except Exception as e:
                logging.error(f"Error loading texts: {e}")
                return []
        else:
            return []
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Success flag
        """
        if not self.vector_store:
            logging.error("Vector store not initialized")
            return False
        
        if not self.model:
            logging.error("Embedding model not initialized")
            return False
        
        try:
            # Extract text content
            texts = [doc.get("content", "") for doc in documents]
            ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
            metadata = [{"source": doc.get("source", ""), "date": doc.get("date", "")} for doc in documents]
            
            # Generate embeddings
            embeddings = self.model.encode(texts)
            
            # Add to vector store
            if self.vector_store_type == "faiss" and FAISS_AVAILABLE:
                # Add to FAISS index
                self.vector_store["index"].add(np.array(embeddings).astype('float32'))
                # Store texts
                self.vector_store["texts"].extend(texts)
                # Save index
                index_path = Path(self.config.get("faiss_index_path", "data/faiss_index"))
                os.makedirs(index_path.parent, exist_ok=True)
                faiss.write_index(self.vector_store["index"], str(index_path))
                # Save texts
                texts_path = index_path.with_suffix(".txt")
                with open(texts_path, "w", encoding="utf-8") as f:
                    f.write("\n===DOCUMENT_SEPARATOR===\n".join(self.vector_store["texts"]))
            
            elif self.vector_store_type == "pinecone" and PINECONE_AVAILABLE:
                # Convert embeddings to list format
                vectors = [(ids[i], emb.tolist(), metadata[i]) for i, emb in enumerate(embeddings)]
                # Upsert to Pinecone
                self.vector_store["index"].upsert(vectors=vectors)
            
            else:
                # Fallback to dictionary-based store
                self.vector_store["texts"].extend(texts)
                self.vector_store["embeddings"].extend(embeddings)
            
            logging.info(f"Added {len(documents)} documents to vector store")
            return True
        
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {e}")
            return False
    
    async def query(self, query: str, top_k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant documents
        
        Args:
            query: Query string
            top_k: Number of results to return (overrides default)
            threshold: Similarity threshold (overrides default)
            
        Returns:
            List of relevant document dictionaries
        """
        if not self.vector_store:
            logging.error("Vector store not initialized")
            return []
        
        if not self.model:
            logging.error("Embedding model not initialized")
            return []
        
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            # Search vector store
            if self.vector_store_type == "faiss" and FAISS_AVAILABLE:
                # Search FAISS index
                distances, indices = self.vector_store["index"].search(
                    np.array([query_embedding]).astype("float32"), top_k
                )
                
                results = []
                for i, idx in enumerate(indices[0]):
                    distance = distances[0][i]
                    # Convert distance to similarity score (1 - normalized distance)
                    similarity = 1.0 - min(distance / 100.0, 1.0)
                    
                    if similarity >= threshold and idx < len(self.vector_store["texts"]):
                        results.append({
                            "content": self.vector_store["texts"][idx],
                            "similarity": float(similarity),
                            "index": int(idx)
                        })
                
                return results
            
            elif self.vector_store_type == "pinecone" and PINECONE_AVAILABLE:
                # Query Pinecone
                query_result = self.vector_store["index"].query(
                    vector=query_embedding.tolist(),
                    top_k=top_k,
                    include_metadata=True
                )
                
                results = []
                for match in query_result.matches:
                    if match.score >= threshold:
                        results.append({
                            "id": match.id,
                            "content": match.metadata.get("text", ""),
                            "source": match.metadata.get("source", ""),
                            "date": match.metadata.get("date", ""),
                            "similarity": float(match.score)
                        })
                
                return results
            
            else:
                # Fallback to dictionary-based store
                results = []
                for i, emb in enumerate(self.vector_store.get("embeddings", [])):
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                    )
                    
                    if similarity >= threshold:
                        results.append({
                            "content": self.vector_store["texts"][i],
                            "similarity": float(similarity),
                            "index": i
                        })
                
                # Sort by similarity (descending) and limit to top_k
                results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
                return results
        
        except Exception as e:
            logging.error(f"Error querying vector store: {e}")
            return []
    
    async def query_by_keywords(self, keywords: List[str], operator: str = "or", **kwargs) -> List[Dict[str, Any]]:
        """
        Query the vector store using multiple keywords
        
        Args:
            keywords: List of keywords to search for
            operator: Logical operator ('and' or 'or')
            **kwargs: Additional arguments for query method
            
        Returns:
            List of relevant document dictionaries
        """
        if not keywords:
            return []
        
        try:
            if operator.lower() == "and":
                # For AND operation, get results for each keyword and find intersection
                all_results = []
                for keyword in keywords:
                    results = await self.query(keyword, **kwargs)
                    all_results.append(set(result["index"] for result in results))
                
                if not all_results:
                    return []
                
                # Find indices present in all result sets
                common_indices = set.intersection(*all_results)
                
                # Construct final results using the first keyword's results as template
                first_results = await self.query(keywords[0], **kwargs)
                return [result for result in first_results if result["index"] in common_indices]
            
            else:  # Default to OR
                # For OR operation, combine results from all keywords
                all_results = {}
                for keyword in keywords:
                    results = await self.query(keyword, **kwargs)
                    for result in results:
                        idx = result["index"]
                        if idx not in all_results or result["similarity"] > all_results[idx]["similarity"]:
                            all_results[idx] = result
                
                # Convert dict to list and sort by similarity
                combined_results = list(all_results.values())
                return sorted(combined_results, key=lambda x: x["similarity"], reverse=True)
        
        except Exception as e:
            logging.error(f"Error in query_by_keywords: {e}")
            return []
    
    async def delete_document(self, doc_id: Union[str, int]) -> bool:
        """
        Delete a document from the vector store
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Success flag
        """
        if not self.vector_store:
            logging.error("Vector store not initialized")
            return False
        
        try:
            if self.vector_store_type == "pinecone" and PINECONE_AVAILABLE:
                # Delete from Pinecone
                self.vector_store["index"].delete(ids=[str(doc_id)])
                logging.info(f"Deleted document {doc_id} from Pinecone")
                return True
            
            else:
                logging.warning("Delete operation not supported for this vector store type")
                return False
        
        except Exception as e:
            logging.error(f"Error deleting document: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        # No specific cleanup needed for most vector stores
        pass