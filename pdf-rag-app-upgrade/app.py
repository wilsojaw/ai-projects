import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

st.set_page_config(page_title="📄 PDF Q&A App", layout="wide")
st.title("🧠 Ask Questions About Your PDFs")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    with st.spinner("Processing documents..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xl",
            model_kwargs={"temperature": 0.3, "max_length": 512},
            task="text-generation"
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Getting answer..."):
            result = qa(query)
        st.subheader("Answer")
        st.write(result["result"])
        st.subheader("Sources")
        for doc in result["source_documents"]:
            st.markdown(f"**{doc.metadata.get('source', 'Unknown')}**")
            st.write(doc.page_content[:300] + "...")
