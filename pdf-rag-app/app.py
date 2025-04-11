import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables from .env file
load_dotenv()

st.title("📄💬 Multi-PDF Q&A with LangChain + Streamlit")

uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.3, "max_length": 512},
        task="text-generation"
)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    query = st.text_input("Ask a question about your PDFs:")

    if query:
        result = qa(query)
        st.subheader("🧠 Answer")
        st.write(result["result"])

        st.subheader("📄 Sources")
        for doc in result["source_documents"]:
            st.markdown(f"**{doc.metadata.get('source', 'Unknown Source')}**")
            st.write(doc.page_content[:300] + "...")
