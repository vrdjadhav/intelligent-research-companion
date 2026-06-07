# src/chatbot.py
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

# 1. Load environment variables safely
load_dotenv()

# 2. Initialize the modern Gemini client
client = genai.Client()

# 3. Define a strict schema for Type-Safe Routing Outputs
class IntentRouterSchema(BaseModel):
    reasoning: str = Field(description="A brief sentence explaining why this command is global or local.")
    routing_mode: Literal["GLOBAL", "LOCAL"] = Field(description="Must be strictly 'GLOBAL' or 'LOCAL'")

def determine_routing_intent(user_message):
    """
    Type-safe, schema-enforced background evaluation to determine whether 
    the command requires a global full-text sweep or localized FAISS fragments.
    Includes a resilient keyword fallback guard.
    """
    # Defensive Fallback List in case the API drops connection mid-flight
    global_keywords = ["index", "chapters", "summary", "summarize", "overview", "syllabus", "notes", "quiz"]
    cleaned_msg = user_message.lower()

    try:
        router_prompt = (
            "You are a strict backend traffic routing microservice for a document AI system.\n"
            "Analyze the user's command and classify its scope into either 'GLOBAL' or 'LOCAL'.\n\n"
            "CRITERIA:\n"
            "- GLOBAL: The user wants an index, table of contents, full chapter list, a holistic summary of the "
            "whole book, a comprehensive study note guide of everything, or a complete document quiz.\n"
            "- LOCAL: The user is asking about a specific term, definition, single concept, snippet of code, "
            "or particular detail found on a specific page."
        )
        
        # Requesting a structurally guaranteed JSON response matching our Pydantic schema
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{router_prompt}\n\nUser Command: {user_message}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=IntentRouterSchema,
                max_output_tokens=100
            )
        )
        
        # Safely parse the structural object response
        structured_data = IntentRouterSchema.model_validate_json(response.text)
        return structured_data.routing_mode

    except Exception as e:
        print(f"\n[ROUTER API ERROR]: {e}. Activating Resilient Keyword Fallback Guard...")
        # Step 3 Optimization: Fallback search if the API stumbles
        for word in global_keywords:
            if word in cleaned_msg:
                print(f"[FALLBACK SUCCESS]: Found keyword '{word}'. Routing to GLOBAL.")
                return "GLOBAL"
        
        print("[FALLBACK DEFAULT]: No global keys found. Routing to LOCAL.")
        return "LOCAL"

def start_new_chat():
    """
    Initializes a fresh Gemini chat session with an elite, highly sophisticated
    butler persona, completely neutralized of any permanent project names.
    """
    try:
        butler_instructions = (
            "You are a highly sophisticated, articulate, and fiercely loyal AI companion "
            "and professional right-hand man. Address the user with the utmost respect, "
            "using terms like 'Sir' or 'Master'. "
            "Your tone is exceptionally polite, formal, composed, and intellectually sharp. "
            "When helping with documents, study guides, quizzes, or notes, execute them with "
            "immaculate high-standard precision. If context is provided, rely on it completely "
            "while maintaining your refined, high-standard demeanor. Never break character."
        )

        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=butler_instructions,
                temperature=0.4
            )
        )
        return chat
    except Exception as e:
        print(f"\n[CRITICAL] Failed to initialize Chat Session: {e}\n")
        return None

def send_chat_message(chat_session, user_message, context=None, routing_mode="LOCAL"):
    """
    Forwards commands along with context. Implements a Context Truncation Guard
    to handle massive text injections smoothly.
    """
    if chat_session is None:
        return "⚠️ Apologies, Sir. The chat session could not be initialized at this time."
        
    # Step 2 Optimization: Context Truncation Guard (approx. 800k character threshold)
    MAX_CHARACTER_THRESHOLD = 800000
    if context and len(context) > MAX_CHARACTER_THRESHOLD:
        print(f"[GUARDRAIL]: Context string length ({len(context)}) exceeds safety threshold. Truncating.")
        context = context[:MAX_CHARACTER_THRESHOLD] + "\n\n[SYSTEM NOTICE: Context safely truncated due to capacity guardrails.]"
        
    try:
        if context:
            if routing_mode == "GLOBAL":
                full_prompt = (
                    f"Sir, I have bypassed local fragment sorting and executed a holistic global structural scan "
                    f"of the complete document matrix to satisfy your broad command.\n\n"
                    f"=== FULL SYSTEM DOCUMENT MATRIX ===\n{context}\n===================================\n\n"
                    f"User Command: {user_message}"
                )
            else:
                full_prompt = (
                    f"Sir, I have completed a targeted scan of the uploaded document coordinates. "
                    f"Please utilize these specific extracted fragments to address the query.\n\n"
                    f"=== SYSTEM DOCUMENT SCAN ===\n{context}\n============================\n\n"
                    f"User Command: {user_message}"
                )
        else:
            full_prompt = user_message

        response = chat_session.send_message(full_prompt)
        return response.text
        
    except Exception as e:
        print(f"\n[CORE ERROR]: {e}\n")
        return f"⚠️ Forgive me, Sir. An unexpected anomaly occurred while generating the response: {e}"