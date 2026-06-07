# app.py
import streamlit as st
import time  # For giving visual step-by-step assurance delays

# 1. Page Configuration (Clean, Unbranded Layout)
st.set_page_config(
    page_title="Intelligent Research Companion",
    page_icon="🤖",
    layout="centered"
)

# 2. Imports
from src.chatbot import start_new_chat, send_chat_message
from src.pdf_handler import extract_text_from_pdf
from src.text_splitter import split_text_into_chunks
from src.vector_store import create_vector_db, search_vector_db

# 3. Initialize Session States
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = start_new_chat()

if "ui_history" not in st.session_state:
    st.session_state.ui_history = []

if "extracted_pdf_text" not in st.session_state:
    st.session_state.extracted_pdf_text = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "mapped_chunks" not in st.session_state:
    st.session_state.mapped_chunks = []


# App Headers (Neutral Branding)
st.title("🤖 Intelligent Research Companion")
st.caption("Version 5.0: High-Capacity Document Intelligence Engine")


# 4. Sidebar UI: Project Info, PDF Uploader, and Actions
with st.sidebar:
    st.header("Project Info")
    st.write("**Version:** 5.0 (RAG Intel Engine)")
    st.write("**Developer:** Technical Assistant")
    st.write("**Status:** Active Companion")
    
    st.markdown("---")
    
    # File Uploader Component
    uploaded_file = st.file_uploader("Upload a study PDF", type=["pdf"], key="pdf_uploader_widget")
    
    # Upgraded Processing Pipeline Layer with Explicit Progress Proofs
    if uploaded_file is not None:
        if st.session_state.get("current_file_name") != uploaded_file.name:
            # Step 1: Extraction
            with st.spinner("Step 1/3: Scanning document structure, Sir..."):
                text = extract_text_from_pdf(uploaded_file)
                time.sleep(0.6)  # UI hold for assurance proof
                
            if text:
                st.sidebar.success("✅ Step 1/3 Complete: Text Extracted!")
                st.session_state.extracted_pdf_text = text
                st.session_state.current_file_name = uploaded_file.name
                
                # Step 2: Chunking
                with st.spinner("Step 2/3: Segmenting text data matrix into chunks..."):
                    time.sleep(0.5)  # Let user see step 2 loading
                    chunks = split_text_into_chunks(text)
                    time.sleep(0.4)
                st.sidebar.success(f"✅ Step 2/3 Complete: Split into {len(chunks)} fragments!")
                    
                # Step 3: Vector Indexing
                with st.spinner("Step 3/3: Mathematical vectorization & mapping..."):
                    index, valid_chunks = create_vector_db(chunks)
                    time.sleep(0.5)
                    
                if index is not None:
                    st.session_state.faiss_index = index
                    st.session_state.mapped_chunks = valid_chunks
                    st.sidebar.success(f"🧠 Step 3/3 Complete: Local Memory Map Active!")
                else:
                    st.error("❌ Vector database index generation failed, Sir.")
            else:
                st.error("❌ Failed to parse the PDF contents, Sir.")
    else:
        # Soft-wipe tracking dependencies if uploader container gets cleared out
        st.session_state.extracted_pdf_text = None
        st.session_state.current_file_name = None
        st.session_state.faiss_index = None
        st.session_state.mapped_chunks = []
                
    st.markdown("---")
    
    # Reset Controls
    if st.button("🗑️ Clear Chat & Document Matrix", use_container_width=True):
        st.session_state.gemini_chat = start_new_chat()
        st.session_state.ui_history = []  
        st.session_state.extracted_pdf_text = None
        st.session_state.current_file_name = None
        st.session_state.faiss_index = None
        st.session_state.mapped_chunks = []
        
        if "pdf_uploader_widget" in st.session_state:
            del st.session_state["pdf_uploader_widget"]
            
        st.rerun()


# =====================================================================
# MAIN SCREEN LAYOUT (Clean Workspace - Extracted Text Area Completely Removed)
# =====================================================================

# Container to hold chat history elements
chat_container = st.container()

# Step 6: Display Existing Chat History with Recalled Citations
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

# Step 7: Handle New User Input
if user_input := st.chat_input("Enter your command, Sir..."):
    
    # Store user command in UI list immediately
    st.session_state.ui_history.append({"role": "user", "text": user_input, "citations": None})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            context_payload = None
            matched_fragments = []
            
            # 1. Scan FAISS index for relevant segments
            if st.session_state.faiss_index is not None:
                with st.spinner("Scanning localized document coordinates, Sir..."):
                    matched_fragments = search_vector_db(
                        index=st.session_state.faiss_index,
                        valid_chunks=st.session_state.mapped_chunks,
                        query=user_input,
                        top_k=3
                    )
                    if matched_fragments:
                        context_payload = "\n\n---\n\n".join(matched_fragments)

            # 2. Get answer from backend companion
            with st.spinner("Processing request, Sir..."):
                response_text = send_chat_message(
                    st.session_state.gemini_chat, 
                    user_input, 
                    context=context_payload
                )
                st.markdown(response_text)
                
                # 3. Present the Source Citations Expander directly below the response
                if matched_fragments:
                    with st.expander("🔍 System Scan: View Retrieved Source Fragments"):
                        for idx, chunk in enumerate(matched_fragments, 1):
                            st.caption(f"**Fragment Vector Match #{idx}:**")
                            st.write(chunk)
                            st.markdown("---")
                
                # Cache response and citations into history tracking list
                st.session_state.ui_history.append({
                    "role": "assistant", 
                    "text": response_text,
                    "citations": matched_fragments if matched_fragments else None
                })
                
    st.rerun()