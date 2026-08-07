# Enterprise RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that lets you chat with your own PDF documents securely and accurately — built to demonstrate context window management, hallucination prevention, document chunking, and embedding-based retrieval.

## Overview

This system ingests PDF documents, breaks them into searchable chunks, converts those chunks into vector embeddings, and stores them in a local vector database. When a user asks a question, the system retrieves the most relevant chunks and passes them to an LLM (Google Gemini) to generate a grounded, accurate answer — one that explicitly avoids making up information not found in the source documents.

## Architecture

```
PDF documents (data/)
        │
        ▼
Document Loader (PyPDFDirectoryLoader)
        │
        ▼
Text Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
Embeddings (HuggingFace all-MiniLM-L6-v2, local/free)
        │
        ▼
Vector Store (FAISS, stored locally)
        │
        ▼
Retriever (top-k similarity search)
        │
        ▼
LLM Generation (Google Gemini) — answers ONLY from retrieved context
        │
        ▼
Streamlit Chat Interface
```

## Tech Stack

- **Python 3.12**
- **LangChain** — orchestration, chunking, retrieval chains
- **FAISS** — local vector database (chosen over ChromaDB to avoid requiring a C++ compiler on Windows)
- **HuggingFace Sentence Transformers** (`all-MiniLM-L6-v2`) — free, local embeddings
- **Google Gemini API** (`gemini-flash-latest`) — LLM generation
- **Streamlit** — chat interface

## Key Features

- **Hallucination prevention**: the system prompt explicitly instructs the LLM to say "the information is not found in company records" rather than guessing, when a question falls outside the indexed documents.
- **Source-grounded answers**: retrieval always pulls from the actual indexed PDFs — answers are never generated from the model's general knowledge alone.
- **Local, cost-free embeddings**: no per-query embedding API costs, since embeddings run locally via sentence-transformers.
- **Re-indexable**: the sidebar "Index/Re-index System Data" button lets you rebuild the vector store any time you add or change documents in the `data/` folder.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/Rejoice-hub/enterprise-rag-pipeline.git
   cd enterprise-rag-pipeline
   ```

2. Create and activate a virtual environment (Python 3.12 recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Google Gemini API key:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

5. Add one or more PDF files to the `data/` folder.

6. Run the app:
   ```
   streamlit run app.py
   ```

7. In the browser, click **"Index/Re-index System Data"** in the sidebar, then start chatting.

## Notes on Engineering Decisions

- **FAISS over Chroma**: Chroma's dependency (`chroma-hnswlib`) requires compiling native code, which needs Microsoft C++ Build Tools on Windows. FAISS ships with pre-built wheels, avoiding that setup friction entirely — a practical trade-off for portability.
- **Text sanitization**: PDF text extraction can produce corrupted/invalid Unicode characters that break tokenization. The ingestion pipeline strips these before embedding to prevent crashes.
- **Model versioning**: LLM provider model names change over time (e.g., `gemini-1.5-flash` was deprecated). Using `gemini-flash-latest` avoids hardcoding a model version that may be retired.

## Future Improvements

- Add support for additional data sources (Notion, SQL databases) as described in the original project scope
- Add citation/source display so answers show which document/page they came from
- Build a small evaluation set to measure retrieval accuracy and hallucination rate
- Add metadata filtering to scope retrieval to specific document categories
