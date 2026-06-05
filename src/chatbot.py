# src/chatbot.py

import google.generativeai as genai

def start_new_chat():
    """
    Initializes a fresh, clean Gemini chat session.
    Configures the model variant and sets up backend communication rules.
    """
    try:
        # Using gemini-1.5-flash as our fast, responsive study companion
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=[])
        return chat
    except Exception as e:
        print(f"\n[CRITICAL] Failed to initialize Gemini Chat Session: {e}\n")
        return None

def send_chat_message(chat_session, user_message, context=None):
    """
    Sends a user message to an active chat session with optional document context.
    
    Optimized to prevent 'History Ballooning' by structuring the context turn
    cleanly without spamming repetitive system headers into the chat history.
    """
    if chat_session is None:
        return "⚠️ Chat session could not be initialized."
        
    try:
        # If document text exists, wrap it neatly so Gemini treats it as active context
        if context:
            full_prompt = (
                f"Context from uploaded document:\n{context}\n\n"
                f"Question: {user_message}"
            )
        else:
            # Fall back to standard messaging if no document is active
            full_prompt = user_message

        # Send the payload to the active session tracking backend
        response = chat_session.send_message(full_prompt)
        return response.text
        
    except Exception as e:
        # Prints the precise API or formatting exception directly to your running terminal
        print(f"\n[GEMINI API ERROR]: {e}\n")
        return "⚠️ An error occurred while generating the response."