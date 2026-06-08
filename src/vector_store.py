# src/vector_store.py
import os
import time
import json
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# Aligned with gemini-embedding-001's structural architecture output
VECTOR_DIMENSION = 3072  
CACHE_DIR = "vector_cache"
INDEX_FILE = os.path.join(CACHE_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(CACHE_DIR, "chunks_mapping.json")

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
    """Generates embedding vectors in safe batches and appends them to the multi-document pool."""
    if not new_chunks:
        return existing_index, existing_chunks or []

    try:
        all_embeddings = []
        batch_size = 100  
        
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch
            )
            batch_vectors = [img.values for img in response.embeddings]
            all_embeddings.extend(batch_vectors)
            time.sleep(0.1)
            
        new_embeddings_np = np.array(all_embeddings).astype('float32')
        
        if existing_index is None:
            index = faiss.IndexFlatL2(VECTOR_DIMENSION)
            combined_chunks = list(new_chunks)
        else:
            index = existing_index
            combined_chunks = existing_chunks + list(new_chunks)
            
        index.add(new_embeddings_np)
        
        # Auto-serialize to disk every time new documents are added!
        save_vector_db(index, combined_chunks)
        
        return index, combined_chunks
        
    except Exception as e:
        print(f"[VECTOR DATABASE ERROR]: {e}")
        return existing_index, existing_chunks or []

def save_vector_db(index, chunks):
    """Serializes the FAISS index and plain-text chunk strings directly to the local hard drive cache."""
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
        # Save the FAISS binary matrix
        faiss.write_index(index, INDEX_FILE)
        
        # Save the accompanying text chunks as JSON
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)
            
        print("💾 [DISK CACHE]: Vector index and chunk matrices securely serialized to disk.")
    except Exception as e:
        print(f"❌ Error serializing vector database to disk: {e}")

def load_vector_db():
    """Loads the cached FAISS index and chunks from the local hard drive back into RAM."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        print("ℹ️ [DISK CACHE]: No local vector serialization found. Starting fresh.")
        return None, []
        
    try:
        # Read the binary index matrix
        index = faiss.read_index(INDEX_FILE)
        
        # Read the text chunks
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        print("⚡ [DISK CACHE]: Local FAISS index and chunk maps hotloaded back into active memory successfully.")
        return index, chunks
    except Exception as e:
        print(f"❌ Error loading serialized vector database: {e}")
        return None, []

def clear_local_cache():
    """Wipes the local cache files from the hard drive."""
    try:
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(CHUNKS_FILE):
            os.remove(CHUNKS_FILE)
        print("🗑️ [DISK CACHE]: Local vector cache completely purged.")
    except Exception as e:
        print(f"❌ Error purging local cache: {e}")

def search_vector_db(index, valid_chunks, query, top_k=3):
    """Queries the multi-document vector array for relevant context fragments."""
    if index is None or not valid_chunks:
        return []
        
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query
        )
        query_embedding = np.array([response.embeddings[0].values]).astype('float32')
        
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