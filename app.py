# app.py
import streamlit as st

# 1. Page Configuration (Preserved Varad AI Layout Layout)
st.set_page_config(
    page_title="Varad AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# 2. Lazy Imports to prevent early runtime threading issues
from src.chatbot import start_new_chat, send_chat_message
from src.pdf_handler import extract_text_from_pdf
from src.text_splitter import split_text_into_chunks
from src.vector_store import create_vector_db, search_vector_db

# 3. Initialize Session States (Preserving original UI tracking names)
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = start_new_chat()

if "ui_history" not in st.session_state:
    st.session_state.ui_history = []

if "extracted_pdf_text" not in st.session_state:
    st.session_state.extracted_pdf_text = None

# Local RAG FAISS Cache hooks
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "mapped_chunks" not in st.session_state:
    st.session_state.mapped_chunks = []


# App Headers
st.title("🤖 Varad AI Assistant")
st.caption("Version 3: High-Capacity Local RAG Companion")


# 4. Sidebar UI: Project Info, PDF Uploader, and Actions
with st.sidebar:
    st.header("Project Info")
    st.write("**Version:** 3.0 (FAISS RAG Engine)")
    st.write("**Developer:** Varad")
    
    st.markdown("---")
    
    # 1. FIXED: Added 'key="pdf_uploader_widget"' to the file uploader component
    uploaded_file = st.file_uploader("Upload a study PDF", type=["pdf"], key="pdf_uploader_widget")
    
    # Rerun-safe Processing Pipeline Layer
    if uploaded_file is not None:
        if st.session_state.get("current_file_name") != uploaded_file.name:
            with st.spinner("Step 1/3: Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                
            if text:
                st.session_state.extracted_pdf_text = text
                st.session_state.current_file_name = uploaded_file.name
                
                with st.spinner("Step 2/3: Slicing into overlapping segments..."):
                    chunks = split_text_into_chunks(text)
                    
                with st.spinner("Step 3/3: Math vectorizing & building local database..."):
                    index, valid_chunks = create_vector_db(chunks)
                    
                if index is not None:
                    # Save FAISS structures into the running tracking scope
                    st.session_state.faiss_index = index
                    st.session_state.mapped_chunks = valid_chunks
                    st.success(f"✅ Indexed {index.ntotal} vector fragments!")
                else:
                    st.error("❌ Vector database index generation failed.")
            else:
                st.error("❌ Failed to read PDF contents.")
    else:
        # Soft-wipe tracking dependencies if uploader container gets cleared out
        st.session_state.extracted_pdf_text = None
        st.session_state.current_file_name = None
        st.session_state.faiss_index = None
        st.session_state.mapped_chunks = []
                
    st.markdown("---")
    
    # Preserved Reset Controls
    if st.button("🗑️ Clear Chat & Document", use_container_width=True):
        st.session_state.gemini_chat = start_new_chat()
        st.session_state.ui_history = []  
        st.session_state.extracted_pdf_text = None
        st.session_state.current_file_name = None
        st.session_state.faiss_index = None
        st.session_state.mapped_chunks = []
        
        # 2. FIXED: Wipe out the internal memory of the file uploader widget
        if "pdf_uploader_widget" in st.session_state:
            del st.session_state["pdf_uploader_widget"]
            
        st.rerun()


# =====================================================================
# MAIN SCREEN LAYOUT
# =====================================================================

# Step 5: Document Preview Section (Preserved)
if st.session_state.extracted_pdf_text:
    st.markdown("### 📄 Document Control Center")
    
    show_text = st.checkbox("👁️ Show Document Extracted Text", value=True)
    
    if show_text:
        with st.container(border=True):
            st.caption(f"Showing preview of the extracted document ({len(st.session_state.extracted_pdf_text)} characters):")
            st.text_area(
                label="Extracted Content", 
                value=st.session_state.extracted_pdf_text, 
                height=200, 
                disabled=True,
                label_visibility="collapsed"
            )
            # Real-time vector memory visibility tracking box
            if st.session_state.faiss_index is not None:
                st.info(f"🧠 Local FAISS memory tracking: **{st.session_state.faiss_index.ntotal}** semantic fragments.")
    st.markdown("---")

# Container to isolate structural chat elements
chat_container = st.container()

# Step 6: Render active UI History stream
with chat_container:
    for message in st.session_state.ui_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

# Step 7: Handle incoming user input
if user_input := st.chat_input("Ask me anything about your document..."):
    
    # Push immediate presentation response 
    st.session_state.ui_history.append({"role": "user", "text": user_input})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            # --- TARGETED CONTEXT SCAN ---
            context_payload = None
            
            if st.session_state.faiss_index is not None:
                with st.spinner("Searching local document database index..."):
                    # Extract the top 3 nodes closest in conceptual space to the user query
                    matched_fragments = search_vector_db(
                        index=st.session_state.faiss_index,
                        valid_chunks=st.session_state.mapped_chunks,
                        query=user_input,
                        top_k=3
                    )
                    if matched_fragments:
                        context_payload = "\n\n---\n\n".join(matched_fragments)

            with st.spinner("Thinking..."):
                # Forward query via our secure isolated chat loop execution routine
                response_text = send_chat_message(
                    st.session_state.gemini_chat, 
                    user_input, 
                    context=context_payload
                )
                st.markdown(response_text)
                
                # Append finalized outcome back to UI visibility tracks
                st.session_state.ui_history.append({"role": "assistant", "text": response_text})
                
    st.rerun()