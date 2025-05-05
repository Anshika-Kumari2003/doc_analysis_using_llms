import os
from typing import Dict, List
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("api_key")
INDEX_NAME = "document-analysis"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Embedding model

def init_pinecone_and_embeddings():
    """Initialize Pinecone and embedding model"""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists(MODELS_DIR):
            print(f"Creating models directory at {MODELS_DIR}...")
            os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Check if embedding model exists
        embedding_model_path = os.path.join(MODELS_DIR, "embedding_model")
        if os.path.exists(embedding_model_path):
            print("Loading embedding model from disk...")
            embeddings_model = SentenceTransformer(embedding_model_path)
        else:
            print("Embedding model not found locally, downloading...")
            embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Save the embedding model for future use
            try:
                print(f"Saving embedding model to {embedding_model_path}...")
                embeddings_model.save(embedding_model_path)
                print("Embedding model saved successfully!")
            except Exception as e:
                print(f"Warning: Failed to save embedding model: {e}")
                # Continue without saving
    except Exception as e:
        print(f"Warning: Error with models directory: {e}")
        print("Continuing with in-memory embedding model only...")
        embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Models loaded successfully!")
    return pc, index, embeddings_model

def semantic_search(index, embeddings_model, query, namespace, top_k=3, score_threshold=0.4, max_results=2):
    """
    Perform semantic search with score filtering and result limiting.

    Args:
        index: Pinecone index object
        embeddings_model: Embedding model
        query: Query string
        namespace: Namespace to search within
        top_k: Number of candidates to retrieve from Pinecone
        score_threshold: Minimum similarity score
        max_results: Max results to return after filtering

    Returns:
        Dictionary of results grouped by document_id
    """
    query_embedding = embeddings_model.encode(query, convert_to_numpy=True).tolist()

    response = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=top_k,
        include_metadata=True
    )

    # Filter and sort matches
    filtered_matches = sorted(
        [m for m in response.get("matches", []) if m["score"] >= score_threshold],
        key=lambda x: x["score"],
        reverse=True
    )[:max_results]

    # Group by document_id
    grouped_results = defaultdict(list)
    for match in filtered_matches:
        metadata = match.get("metadata", {})
        doc_id = metadata.get("document_id", f"unknown-{match['id']}")
        grouped_results[doc_id].append({
            "score": match["score"],
            "text": metadata.get("text", "No preview available")
        })

    return grouped_results

def format_documents(results_by_doc: Dict) -> str:
    """Format retrieved documents for inclusion in the context."""
    if not results_by_doc:
        return "No relevant information found."
    
    formatted_text = ""
    for doc_id, matches in results_by_doc.items():
        for match in matches:
            formatted_text += f"\n--- From document: {doc_id} (score: {match['score']:.2f}) ---\n"
            formatted_text += match["text"].strip() + "\n"
    
    return formatted_text

def process_query(index, embeddings_model, company: str, query: str) -> Dict:
    """
    Process a query for a specific company and return relevant document chunks.
    
    Args:
        index: Pinecone index object
        embeddings_model: Embedding model
        company: Company namespace to search
        query: Query string
        
    Returns:
        Dictionary of search results grouped by document_id
    """
    if not query.strip():
        return {}
    
    # Search Pinecone for relevant context
    results = semantic_search(
        index=index,
        embeddings_model=embeddings_model,
        query=query,
        namespace=company.lower(),
        top_k=4,
        score_threshold=0.4,
        max_results=3
    )
    
    return results 