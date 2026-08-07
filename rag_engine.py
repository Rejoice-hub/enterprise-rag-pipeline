import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_DIR = "./faiss_db"
DATA_DIR = "./data"

# Initialize Free Local Embeddings and Free Gemini LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

def ingest_docs_to_vector_store():
    """Reads PDFs from data folder, chunks them, and stores them in FAISS."""
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        return None

    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    # Chunking data to fit into the LLM context window
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_chunks = text_splitter.split_documents(docs)

    # Filter out empty/invalid chunks and strip corrupted unicode characters
    cleaned_chunks = []
    for chunk in final_chunks:
        if isinstance(chunk.page_content, str) and chunk.page_content.strip():
            clean_text = chunk.page_content.encode("utf-8", errors="ignore").decode("utf-8")
            clean_text = clean_text.replace("\x00", "")
            if clean_text.strip():
                chunk.page_content = clean_text
                cleaned_chunks.append(chunk)

    final_chunks = cleaned_chunks

    if not final_chunks:
        return None

    print(f"Number of chunks: {len(final_chunks)}")

    # Store vectors locally
    vector_store = FAISS.from_documents(final_chunks, embeddings)
    vector_store.save_local(DB_DIR)
    return vector_store

def get_rag_chain():
    """Creates the automated question-answering system blueprint."""
    vector_store = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "You are an expert corporate AI assistant. Use the following pieces of retrieved "
        "context to answer the user question. If you don't know the answer, say clearly that "
        "the information is not found in company records. Do not make up information.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)