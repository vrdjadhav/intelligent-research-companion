# 🤖 Intelligent Research Companion

An AI-powered learning assistant built to explore AI Engineering, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Agentic AI through hands-on development.

## 🎯 Vision

Build a learning companion that helps students understand study materials, interact with documents, generate learning resources, and eventually automate parts of the learning process through intelligent AI workflows.

---

## ✨ Current Features

### 💬 AI Chat Assistant & Elite Persona
* Chat with Google's advanced Gemini model layered under a highly sophisticated, formal butler/right-hand-man persona.
* Maintain fully stateful conversation history with multi-turn capabilities.
* Persistent clear chat and computational matrix flush functionality.

### 🧠 LLM-Driven Dynamic Intent Routing
* Automated background intent classification microservice using strict Pydantic JSON schemas.
* Dynamic traffic routing that switches between global document summaries and localized context lookups effortlessly.
* Resilient string-matching keyword fallback guard in case of external API disruptions.

### 🗂️ Advanced Local RAG Execution
* Automated text matrix segmentation using variable chunk sizes with character overlap boundaries.
* High-speed mathematical vector indexing utilizing local FAISS vector stores.
* Interactive UI citations expander showing exact localized document coordinate fragments underneath responses.
* Context Truncation Guard preventing API crashes by safely monitoring character length thresholds.

---

## ⚙️ How It Works

```text
       Upload PDF
            │
    Extract Text Matrix
            │
  Chunking & Vectorization ──► Saved to FAISS Index
            │
    User Enters Command
            │
   [LLM Intent Router] ───────► Evaluates Scope (GLOBAL vs LOCAL)
      ╱            ╲
     ╱ GLOBAL       ╲ LOCAL
    ▼                ▼
Full Raw Text     FAISS Segment
 Context           Coordinates
     ╲              ╱
      ▼            ▼
   Gemini Generates Response
            │
   Polite Butler Formatting
```
## 🛠️ Technologies Used

* **Programming Language:** Python
* **UI Architecture:** Streamlit
* **Core LLM Engine:** Google Gemini API (`google-genai` SDK)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Data Validation:** Pydantic (Type-Safe Schema structural enforcement)
* **Document Engine:** PyMuPDF (PDF Parser)
* **Environment Controls:** python-dotenv

---

## 🚀 Development Progress

### Version 1 ✅
* Basic chatbot interface
* Gemini integration endpoints
* Stateful conversation matrices
* Multi-turn chat history recall

### Version 2 ✅
* Standard PDF upload integration
* Raw PDF text extraction
* Pipeline document processing arrays

### Version 3 ✅
* Deep "Chat with PDF" features
* Context-aware backend injection responses
* Session-state document temporary memory storage

### Version 4 ✅
* Automated high-level document summarization
* Generative multi-choice study quiz engine
* Core concept study notes generation algorithms

### Version 5.0 ✅
* Advanced Production RAG pipeline implementation
* Text matrix overlapping token splitters & embeddings
* Local high-performance FAISS semantic search indexing
* Explicit UI step-by-step pipeline assurance checkpoints

### Version 5.1 ✅
* Intelligent dynamic traffic routing architecture
* Strict Pydantic structural schema output enforcement
* Context truncation safety guardrails & fallback keyword protection
* Integrated elite butler core persona behavior adjustments

### Version 5.2 ✅
* Multi-Document Vector Pools architecture
* Expand database coordinates to support multiple simultaneous PDF uploads
* Dynamic batch-processing ingestion to handle massive textbooks/handbooks without payload errors
* Cross-referencing matrix and source filename citation tracking across all uploaded materials concurrently

### Version 6.0 🚧
* Persistent local disk serialization (saving and loading compiled FAISS indexes from hard drive cache)

### Version 7.0 🔜
* Agentic learning workflows
* Weakness mapping evaluation quizzes & personalized tracking agent

---

## 📚 Learning Objectives

Through this project, I aim to strengthen my practical understanding of:
* Modern AI application system development
* Advanced prompt engineering & persona architecture
* LLM integration techniques & Structured JSON API responses
* Hybrid RAG routing logic and dynamic context injection
* Software architecture principles & pipeline safety guardrails
* Enterprise-level AI engineering development workflows

---

## 👨‍💻 Author

**Varad Jadhav** *B.Tech Artificial Intelligence & Data Science Student* Building production-ready systems and learning through rigorous, hands-on development..
