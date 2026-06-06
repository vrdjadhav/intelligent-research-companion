def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    if not text:
        return []
        
    chunks = []
    start_index = 0
    text_length = len(text)
    
    while start_index < text_length:
        end_index = start_index + chunk_size
        
        chunk = text[start_index:end_index]
        chunks.append(chunk.strip())
        
        start_index += (chunk_size - chunk_overlap)
        
    return chunks