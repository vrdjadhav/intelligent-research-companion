# src/vector_store.py
import os
import time
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# Aligned with gemini-embedding-001's structural architecture output
VECTOR_DIMENSION = 3072  

def get_embedding(text):
    """Generates a single 3072-dimension vector for an intent routing or query string."""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"❌ Error generating embedding layer: {e}")
        return None

def create_vector_db(new_chunks, existing_index=None, existing_chunks=None):
    """
    Generates embedding vectors in small, safe batches to prevent API payload errors,
    then appends them cleanly to the FAISS multi-document pool.
    """
    if not new_chunks:
        return existing_index, existing_chunks or []

    try:
        all_embeddings = []
        batch_size = 100  # Safe structural boundary limit for API payload windows
        
        # 1. Process incoming chunks sequentially in safe batches
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            
            # Request embeddings via the standard gemini-embedding-001 engine
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch
            )
            
            # Extract and collect vectors
            batch_vectors = [img.values for img in response.embeddings]
            all_embeddings.extend(batch_vectors)
            
            # Tiny rest to respect API rate limits during massive book loads
            time.sleep(0.1)
            
        # Convert total compiled list to float32 NumPy standard matrix
        new_embeddings_np = np.array(all_embeddings).astype('float32')
        
        # 2. Combine or expand the running FAISS index structure
        if existing_index is None:
            # Set up a flat L2 index matching our 3072 geometry
            index = faiss.IndexFlatL2(VECTOR_DIMENSION)
            combined_chunks = list(new_chunks)
        else:
            index = existing_index
            combined_chunks = existing_chunks + list(new_chunks)
            
        # Add the full batch-compiled matrix directly to the FAISS database pool
        index.add(new_embeddings_np)
        return index, combined_chunks
        
    except Exception as e:
        print(f"[VECTOR DATABASE ERROR]: {e}")
        return existing_index, existing_chunks or []

def search_vector_db(index, valid_chunks, query, top_k=3):
    """
    Queries the multi-document vector array for relevant context fragments.
    """
    if index is None or not valid_chunks:
        return []
        
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query
        )
        query_embedding = np.array([response.embeddings[0].values]).astype('float32')
        
        # Guard against requesting more items than exist in our pool
        actual_k = min(top_k, len(valid_chunks))
        if actual_k == 0:
            return []
            
        distances, indices = index.search(query_embedding, actual_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(valid_chunks):
                results.append(valid_chunks[idx])
        return results
    except Exception as e:
        print(f"[SEARCH ERROR]: {e}")
        return []