# src/agent_engine.py
import json
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# --- STRUCTURED OUTBOUND SCHEMAS (PYDANTIC MATRICES) ---
class QuizQuestion(BaseModel):
    question_id: int = Field(description="Sequential identifier for the question starting at 1.")
    topic: str = Field(description="The specific core concept or chapter being tested.")
    question_text: str = Field(description="A deep, conceptual question derived from the document context.")
    expected_core_concepts: list[str] = Field(description="Key criteria points required to get full marks for this question.")

class QuizSchema(BaseModel):
    quiz_title: str = Field(description="Clean, thematic title for the assessment profile.")
    questions: list[QuizQuestion] = Field(description="Array of 3 conceptual evaluation criteria queries.")

class AssessmentEvaluation(BaseModel):
    question_id: int
    score_out_of_10: int = Field(description="Grading score from 0 to 10 based on conceptual completeness.")
    strengths: str = Field(description="What the user correctly understood in their response.")
    weaknesses: str = Field(description="Crucial gaps, missing points, or misconceptions in the user's answer.")
    remedial_guidance: str = Field(description="Direct corrective explanation of the concept based on source context.")

class MasterReportSchema(BaseModel):
    total_score_percentage: float = Field(description="Aggregated performance score mapped to a 100% scale.")
    evaluations: list[AssessmentEvaluation]
    primary_knowledge_gap: str = Field(description="The single biggest weakness identified across the user's answers.")
    recommended_focus_topics: list[str] = Field(description="Actionable subject tracks the user must re-study.")


# --- AGENTIC EXECUTION FUNCTIONS ---

def generate_evaluation_quiz(mapped_chunks):
    """Agentic Tool: Scans active vector memory to compile a structured 3-question concept quiz."""
    if not mapped_chunks:
        return None

    # Sample context fragments to give the model conceptual diversity
    sample_context = "\n\n---\n\n".join(mapped_chunks[:15])  # Use top chunks for structural scope
    
    prompt = f"""
    You are an elite Academic Evaluation Agent. Your mission is to analyze the following document context 
    and design a rigorous, 3-question conceptual evaluation quiz. 
    Avoid generic queries; focus on tracking deep structural mechanics or core arguments within the text.
    
    SOURCE DOCUMENTS DATA MATRIX:
    {sample_context}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': QuizSchema,
                'temperature': 0.3
            }
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ [QUIZ ENGINE ERROR]: Failed to compile agentic quiz matrix: {e}")
        return None


def evaluate_quiz_submission(quiz_data, user_answers, mapped_chunks):
    """Evaluator Critic Agent: Conceptually scores answers against ground truth source vectors."""
    if not quiz_data or not user_answers:
        return None
        
    master_context = "\n\n---\n\n".join(mapped_chunks[:20])
    
    evaluation_payload = {
        "original_quiz": quiz_data,
        "user_submitted_answers": user_answers
    }

    prompt = f"""
    You are an expert Critic and Remedial Academic Agent. Your assignment is to thoroughly evaluate the 
    user's submitted answers against the original questions and the verified source documentation matrix.
    
    Calculate a meticulous score from 0-10 for each individual answer based on conceptual accuracy and completeness.
    Formulate highly descriptive strength, weakness, and remedial guidance strings for each response.
    
    VERIFIED GROUND TRUTH SOURCE MEMORY:
    {master_context}
    
    EVALUATION TARGET MATRIX (QUIZ & USER INPUT):
    {json.dumps(evaluation_payload, indent=2)}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': MasterReportSchema,
                'temperature': 0.2
            }
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ [CRITIC ENGINE ERROR]: Failed to execute evaluation matrix processing: {e}")
        return None