import streamlit as st

st.set_page_config(
    page_title="Varad AI Assistant",
    page_icon="🤖",
    layout="centered"
)


from src.chatbot import start_new_chat, send_chat_message


if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = start_new_chat()

st.title("🤖 Varad AI Assistant")

with st.sidebar:
    st.header("About")
    st.write("Version 2: RAG-Ready Stateful Chatbot")
    
    if st.button("Clear Chat History"):
        st.session_state.gemini_chat = start_new_chat()
        st.rerun()

if st.session_state.gemini_chat is not None:
    for message in st.session_state.gemini_chat.get_history():
        role = "assistant" if message.role == "model" else message.role
        with st.chat_message(role):
            st.markdown(message.parts[0].text)
else:
    st.error("❌ Failed to connect to Gemini backend. Please check your API key in the .env file and restart the app.")

if user_input := st.chat_input("Ask me anything..."):
    
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = send_chat_message(st.session_state.gemini_chat, user_input)
            st.markdown(response_text)