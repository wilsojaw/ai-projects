# 🧠 PDF RAG App with Streamlit + LangChain

This is a multi-PDF Q&A app built with [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io), and [Hugging Face](https://huggingface.co/).

## Features
- Upload **one or more PDFs**
- Ask **natural language questions**
- Get **smart answers with source references**
- Powered by:
  - [LangChain](https://www.langchain.com/)
  - [FAISS](https://github.com/facebookresearch/faiss)
  - [Hugging Face Transformers](https://huggingface.co/)
  - [Streamlit](https://streamlit.io/)

## Tech Stack
- Python
- LangChain
- Streamlit
- Hugging Face Hub (Flan-T5 model)
- FAISS for vector search
- HuggingFaceEmbeddings (MiniLM-L6-v2)

## How It Works

1. PDFs are loaded and split into chunks.
2. Each chunk is embedded using `all-MiniLM-L6-v2`.
3. FAISS indexes the chunks for fast semantic retrieval.
4. A `flan-t5-large` model answers your question based on the top `k=3` relevant chunks.

## How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/wilsonjaw/pdf-rag-app.git
   cd pdf-rag-app

2. Set up a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
    pip install -r requirements.txt
    
4. Add your .env file
    HUGGINGFACEHUB_API_TOKEN=your_token

5. Run the app 
    streamlit run app.py
