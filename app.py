# app.py
import streamlit as st
import time  # For giving visual step-by-step assurance delays

# 1. Page Configuration (Clean, Unbranded Layout)
st.set_page_config(
    page_title="Intelligent Research Companion",
    page_icon="🤖",
    layout="centered"
)

# 2. Imports (Aligned exactly with your src/pdf_handler.py)
from src.chatbot import start_new_chat, send_chat_message, determine_routing_intent
from src.pdf_handler import extract_text_from_pdf
from src.text_splitter import split_text_into_chunks
from src.vector_store import create_vector_db, search_vector_db

# 3. Initialize Multi-Document Session States (Version 5.2 Architecture)
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = start_new_chat()

if "ui_history" not in st.session_state:
    st.session_state.ui_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # Tracks unique filenames

if "global_text_pool" not in st.session_state:
    st.session_state.global_text_pool = ""     # Combined text matrix for GLOBAL summary mode

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None        # Shared cumulative FAISS index pool

if "mapped_chunks" not in st.session_state:
    st.session_state.mapped_chunks = []        # Shared cumulative text chunks array


# App Headers (Neutral Branding)
st.title("🤖 Intelligent Research Companion")
st.caption("Version 5.2: Multi-Document Vector Pool & Routing Engine")


# 4. Sidebar UI: Project Info, PDF Uploader, and Actions
with st.sidebar:
    st.header("Project Info")
    st.write("**Version:** 5.2 (Multi-Doc RAG Pool)")
    st.write("**Developer:** Technical Assistant")
    st.write("**Status:** Active Companion")
    
    st.markdown("---")
    
    # Configured to accept multiple files simultaneously
    uploaded_files = st.file_uploader(
        "Upload Study Documents (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True, 
        key="pdf_uploader_widget"
    )
    
    # 5. Incremental Processing Pipeline Engine with Step Assurance Proofs
    if uploaded_files:
        for file in uploaded_files:
            # Only run processing if the file has not yet been processed in this session
            if file.name not in st.session_state.processed_files:
                st.markdown(f"### 📄 Processing: {file.name}")
                
                # Step 1: Extraction
                with st.spinner("Step 1/3: Scanning document structure, Sir..."):
                    text = extract_text_from_pdf(file)
                    time.sleep(0.6)  # UI hold for assurance proof
                    
                if text:
                    st.sidebar.success(f"✅ Extracted text matrix from {file.name}!")
                    # Append file contents to global full text pool for global analytics
                    st.session_state.global_text_pool += f"\n\n--- DOCUMENT START: {file.name} ---\n{text}\n--- DOCUMENT END: {file.name} ---"
                    
                    # Step 2: Chunking & Tagging
                    with st.spinner("Step 2/3: Segmenting text data matrix into chunks..."):
                        time.sleep(0.5)
                        raw_chunks = split_text_into_chunks(text)
                        # Tag individual text fragments with their specific file name origins
                        tagged_chunks = [f"[{file.name}]: {chunk}" for chunk in raw_chunks]
                        time.sleep(0.4)
                    st.sidebar.success(f"✅ Split into {len(raw_chunks)} fragments!")
                        
                    # Step 3: Vector Indexing (Appending incrementally to the vector store)
                    with st.spinner("Step 3/3: Mathematical vectorization & pool mapping..."):
                        index, valid_chunks = create_vector_db(
                            new_chunks=tagged_chunks,
                            existing_index=st.session_state.faiss_index,
                            existing_chunks=st.session_state.mapped_chunks
                        )
                        time.sleep(0.5)
                        
                    if index is not None:
                        st.session_state.faiss_index = index
                        st.session_state.mapped_chunks = valid_chunks
                        st.session_state.processed_files.add(file.name)
                        st.sidebar.success(f"🧠 Added to local core memory mapping matrix!")
                    else:
                        st.error(f"❌ Vector database compilation failed for {file.name}, Sir.")
                else:
                    st.error(f"❌ Failed to parse data from {file.name}, Sir.")
                st.markdown("---")
                
    st.markdown("---")
    
    # Reset Controls
    if st.button("🗑️ Clear Chat & Document Matrix", use_container_width=True):
        st.session_state.gemini_chat = start_new_chat()
        st.session_state.ui_history = []  
        st.session_state.processed_files = set()
        st.session_state.global_text_pool = ""
        st.session_state.faiss_index = None
        st.session_state.mapped_chunks = []
        
        if "pdf_uploader_widget" in st.session_state:
            del st.session_state["pdf_uploader_widget"]
            
        st.rerun()


# =====================================================================
# MAIN SCREEN LAYOUT
# =====================================================================

# Container to hold chat history elements
chat_container = st.container()

# Display Existing Chat History with Recalled Citations
with chat_container:
    for message in st.session_state.ui_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            
            # Re-render citation expander blocks if they belong to past responses
            if message.get("role") == "assistant" and message.get("citations"):
                with st.expander("🔍 System Scan: View Retrieved Source Fragments"):
                    for idx, chunk in enumerate(message["citations"], 1):
                        st.caption(f"**Fragment Vector Match #{idx}:**")
                        st.write(chunk)
                        st.markdown("---")

# Handle New User Input
if user_input := st.chat_input("Enter your command, Sir..."):
    
    # Store user command in UI list immediately
    st.session_state.ui_history.append({"role": "user", "text": user_input, "citations": None})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            context_payload = None
            matched_fragments = []
            
            # Step A: Evaluate Traffic Intent Router
            routing_mode = "LOCAL"
            if st.session_state.faiss_index is not None:
                with st.spinner("Analyzing intent profile matrix, Sir..."):
                    routing_mode = determine_routing_intent(user_input)
            
            # Step B: Assemble Context Based On Intended Route
            if st.session_state.faiss_index is not None:
                if routing_mode == "GLOBAL":
                    context_payload = st.session_state.global_text_pool
                else:
                    with st.spinner("Scanning localized document coordinates, Sir..."):
                        matched_fragments = search_vector_db(
                            index=st.session_state.faiss_index,
                            valid_chunks=st.session_state.mapped_chunks,
                            query=user_input,
                            top_k=3
                        )
                        if matched_fragments:
                            context_payload = "\n\n---\n\n".join(matched_fragments)

            # Step C: Dispatch Request to Backend Engine
            with st.spinner("Processing request, Sir..."):
                response_text = send_chat_message(
                    chat_session=st.session_state.gemini_chat, 
                    user_message=user_input, 
                    context=context_payload,
                    routing_mode=routing_mode
                )
                st.markdown(response_text)
                
                # Step D: Present Source Citations Expander (Only during Local Search route matches)
                if routing_mode == "LOCAL" and matched_fragments:
                    with st.expander("🔍 System Scan: View Retrieved Source Fragments"):
                        for idx, chunk in enumerate(matched_fragments, 1):
                            st.caption(f"**Fragment Vector Match #{idx}:**")
                            st.write(chunk)
                            st.markdown("---")
                
                # Cache response and citations into history tracking list
                st.session_state.ui_history.append({
                    "role": "assistant", 
                    "text": response_text,
                    "citations": matched_fragments if (routing_mode == "LOCAL" and matched_fragments) else None
                })
                
    st.rerun()