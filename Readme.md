# RAG Chatbot Model Comparison – Final Submission

This repository contains four implementations of a Retrieval-Augmented Generation (RAG) chatbot system, each using a different combination of large language model (LLM) and embedding model. The chatbot allows users to upload financial PDF documents (such as SEC 10-K reports) and ask questions based strictly on the document content.

We designed this project to systematically compare the performance, behavior, and limitations of various model configurations, while also improving the user experience through prompt engineering and system-level safeguards.

---

## Model Variants and Improvements

### 1. RAG-gemini-gemini.py
- **LLM**: Gemini 2.0 Flash (via Google Generative AI API)
- **Embedding Model**: Gemini Embeddings (`models/embedding-001`)
- **Key Improvements**:
  - Added a role-specific persona tailored to financial analysis of 10-K reports
  - Implemented chat history tracking to support multi-turn dialogue
  - Introduced hallucination fallback logic to prevent unsupported responses
  - Enabled dynamic clarification prompting based on user intent

### 2. RAG-ollama-gemini.py
- **LLM**: Local Ollama model (`mistral`)
- **Embedding Model**: Gemini Embeddings
- **Key Improvements**:
  - Combined local model inference with cloud-based embedding for hybrid deployment
  - Integrated the same persona and fallback logic for consistency
  - Enabled chat history and context-aware prompting across turns

### 3. RAG-openai-openai.py
- **LLM**: OpenAI GPT (`gpt-4`)
- **Embedding Model**: OpenAI Embeddings
- **Key Improvements**:
  - Adopted the same structured prompt template and persona as other variants
  - Implemented chat history and clarification logic
  - Standardized the fallback response mechanism for handling uncertainty

### 4. RAG-ollama-openai.py
- **LLM**: Local Ollama model
- **Embedding Model**: OpenAI Embeddings
- **Key Improvements**:
  - Offline generation combined with cloud-based document retrieval
  - Preserved persona behavior, context handling, and robust error fallback

---

## Summary of Common Enhancements

All four chatbot variants include the following shared enhancements:

- A consistent, custom-designed persona that defines assistant behavior, tone, scope of knowledge, and response boundaries.
- Integration of multi-turn chat history to preserve context across questions and answers.
- Clarification prompting logic that detects vague or unsupported queries and asks users to provide more details.
- A hallucination detection fallback mechanism that ensures the chatbot only responds when sufficient document-grounded evidence is found.
- Support for PDF file uploading, document chunking, embedding via FAISS, and real-time question answering through Streamlit.