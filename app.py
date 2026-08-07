import streamlit as st
import os
from rag_engine import ingest_docs_to_vector_store, get_rag_chain

st.set_page_config(page_title="Enterprise RAG System", layout="wide")
st.title("💼 Enterprise AI Knowledge Base (RAG)")

# Sidebar administration tasks
with st.sidebar:
    st.header("Admin Controls")
    if st.button("🔄 Index/Re-index System Data"):
        with st.spinner("Processing company documents..."):
            db = ingest_docs_to_vector_store()
            if db:
                st.success("Successfully vectorized internal documents!")
            else:
                st.error("Please add some PDF files to the 'data/' directory first.")

st.subheader("Chat with your internal database")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about company policy or data..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching internal data pipelines..."):
            try:
                rag_chain = get_rag_chain()
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Could not complete the query: {e}")