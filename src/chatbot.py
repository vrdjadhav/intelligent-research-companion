from google import genai

from src.config import GEMINI_API_KEY, MODEL_NAME

client = genai.Client(
    api_key = GEMINI_API_KEY
)

def start_new_chat():
    try:
        return client.chats.create(model=MODEL_NAME)
    except Exception as e :
        print(f"Error initializing chat:{e}")
        return None

def send_chat_message(chat_session, user_message):
    if chat_session is None:
        return "⚠️ Chat session could not be initialized. Please check your API key or connection."
        
    try:
        response = chat_session.send_message(user_message)
        return response.text
    
    except Exception as e:
        print(f"Unexpected Backend Error: {e}")
        return