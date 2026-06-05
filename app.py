import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Varad AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# 2. Imports
from src.chatbot import start_new_chat, send_chat_message
from src.pdf_handler import extract_text_from_pdf

# 3. Initialize Session States
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = start_new_chat()

# Keep a clean UI-only history tracking list to prevent prompt leakage on screen
if "ui_history" not in st.session_state:
    st.session_state.ui_history = []

# Cache for the extracted document text
if "extracted_pdf_text" not in st.session_state:
    st.session_state.extracted_pdf_text = None


# App Headers
st.title("🤖 Varad AI Assistant")
st.caption("Version 2: Document-aware learning companion")


# 4. Sidebar UI: Project Info, PDF Uploader, and Actions
with st.sidebar:
    st.header("Project Info")
    st.write("**Version:** 2.0 (PDF Support)")
    st.write("**Developer:** Varad")
    
    st.markdown("---")
    
    # Task 4: File Uploader Component
    uploaded_file = st.file_uploader("Upload a study PDF", type=["pdf"])
    
    # Rerun-safe Extraction Logic
    if uploaded_file is not None:
        # Check if the file currently uploaded is different from what we processed
        # We use the file name to track if it's a new file
        if st.session_state.get("current_file_name") != uploaded_file.name:
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    st.session_state.extracted_pdf_text = text
                    st.session_state.current_file_name = uploaded_file.name
                    st.success("✅ PDF processed successfully!")
                else:
                    st.error("❌ Failed to read PDF.")
    else:
        # If the user clears the file uploader, clear our cached text states
        st.session_state.extracted_pdf_text = None
        st.session_state.current_file_name = None
                
    st.markdown("---")
    
    # Clear Buttons (Resetting everything back to fresh states)
    if st.button("🗑️ Clear Chat & Document", use_container_width=True):
        st.session_state.gemini_chat = start_new_chat()
        st.session_state.ui_history = []  # Clear our clean UI list too!
        st.session_state.extracted_pdf_text = None
        st.session_state.current_file_name = None
        st.rerun()


# =====================================================================
# MAIN SCREEN LAYOUT
# =====================================================================

# Step 5: Document Preview Section (Fixed at the top)
if st.session_state.extracted_pdf_text:
    st.markdown("### 📄 Document Control Center")
    
    # This checkbox acts as an interactive Show/Hide button switch
    show_text = st.checkbox("👁️ Show Document Extracted Text", value=True)
    
    if show_text:
        # Containers give us a clean, bordered box for the text
        with st.container(border=True):
            st.caption(f"Showing preview of the extracted document ({len(st.session_state.extracted_pdf_text)} characters):")
            # Using height keeps it from stretching your whole webpage down!
            st.text_area(
                label="Extracted Content", 
                value=st.session_state.extracted_pdf_text, 
                height=200, 
                disabled=True,
                label_visibility="collapsed"
            )
    st.markdown("---")

# Container to hold chat history so it stays separate from our document top preview
chat_container = st.container()

# Step 6: Display Existing Chat History inside the chat container using our clean UI list
with chat_container:
    for message in st.session_state.ui_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

# Step 7: Handle New User Input
if user_input := st.chat_input("Ask me anything about your document..."):
    
    # 1. Save the clean message to our UI history so it looks right on rerun
    st.session_state.ui_history.append({"role": "user", "text": user_input})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Send the secret mega-prompt payload to the backend
                response_text = send_chat_message(
                    st.session_state.gemini_chat, 
                    user_input, 
                    context=st.session_state.extracted_pdf_text
                )
                st.markdown(response_text)
                
                # 2. Save the assistant's answer to our clean UI history
                st.session_state.ui_history.append({"role": "assistant", "text": response_text})
                
    # Force a clean rerun to refresh the UI layout and state tracking perfectly
    st.rerun()