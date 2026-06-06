# src/chatbot.py
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Load environment variables safely from your local .env file
load_dotenv()

# 2. Initialize the modern Gemini client
client = genai.Client()

def start_new_chat():
    """
    Initializes a fresh, modern Gemini chat session.
    Using gemini-2.5-flash as our fast, multi-turn study companion.
    """
    try:
        # Correct official initiation method for tracking stateless multi-turn chains
        chat = client.chats.create(model="gemini-2.5-flash")
        return chat
    except Exception as e:
        print(f"\n[CRITICAL] Failed to initialize Gemini Chat Session: {e}\n")
        return None

def send_chat_message(chat_session, user_message, context=None):
    """
    Sends a user message to an active chat session with targeted RAG fragments.
    
    Uses standard structural prompts to protect history tracking while 
    leveraging the modern SDK object layer cleanly.
    """
    if chat_session is None:
        return "⚠️ Chat session could not be initialized."
        
    try:
        # If relevant document text fragments exist, present them as temporary instruction boundaries
        if context:
            full_prompt = (
                f"You are a helpful study assistant. Use the following context extracted "
                f"from the document to answer the user's question accurately.\n\n"
                f"=== DOCUMENT CONTEXT ===\n{context}\n========================\n\n"
                f"User Question: {user_message}"
            )
        else:
            full_prompt = user_message

        # Fire the structured text stream payload to the chat backend session
        response = chat_session.send_message(full_prompt)
        return response.text
        
    except Exception as e:
        print(f"\n[GEMINI API ERROR]: {e}\n")
        return f"⚠️ An error occurred while generating the response: {e}"