# src/vector_store.py
import os
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# Defining the exact geometry matching your gemini-embedding-001 vectors
VECTOR_DIMENSION = 3072  

def get_embedding(text):
    """Generates a single 3072-dimension vector for a text string."""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"❌ Error generating embedding layer: {e}")
        return None

def create_vector_db(chunks):
    """
    Takes a list of text chunks, converts them all into vectors,
    and indexes them inside a local FAISS database.
    """
    if not chunks:
        return None, []
    
    print(f"🔄 Vectorizing {len(chunks)} text chunks... Please wait.")
    
    embeddings_list = []
    valid_chunks = []
    
    # 1. Convert every text chunk into its vector counterpart
    for chunk in chunks:
        vector = get_embedding(chunk)
        if vector is not None:
            embeddings_list.append(vector)
            valid_chunks.append(chunk)
            
    if not embeddings_list:
        print("❌ Failsafe triggered: No embeddings could be generated.")
        return None, []

    # 2. Convert raw Python list into a strict Float32 NumPy Matrix (Required by FAISS)
    embeddings_matrix = np.array(embeddings_list).astype('float32')
    
    # 3. Initialize a flat L2 (Euclidean Distance) FAISS Index
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    
    # 4. Add the math matrix directly to the database index
    index.add(embeddings_matrix)
    
    print(f"✅ Successfully indexed {index.ntotal} vectors inside FAISS local database.")
    
    # Return the database index and the matching text array
    return index, valid_chunks
def search_vector_db(index, valid_chunks, query, top_k=2):
    """
    Takes a user question, converts it into an embedding vector,
    and queries the FAISS index to pull out the 'top_k' most relevant text chunks.
    """
    if index is None or not valid_chunks:
        return []
        
    # 1. Convert the user's question into its mathematical coordinates
    query_vector = get_embedding(query)
    if query_vector is None:
        return []
        
    # 2. Reshape the single vector into a strict 2D Float32 NumPy Matrix (Required by FAISS)
    query_matrix = np.array([query_vector]).astype('float32')
    
    # 3. Ask FAISS to find the top_k closest vectors using Euclidean distance
    # 'distances' contains the raw similarity scores, 'indices' contains the position numbers
    distances, indices = index.search(query_matrix, top_k)
    
    # 4. Map those indices back to the actual text strings we saved
    relevant_chunks = []
    for position_idx in indices[0]:
        # FAISS returns -1 if it can't find enough matches
        if position_idx != -1 and position_idx < len(valid_chunks):
            relevant_chunks.append(valid_chunks[position_idx])
            
    return relevant_chunks