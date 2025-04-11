# 🧠 PDF RAG App with Streamlit + LangChain

This is a multi-PDF Q&A app built with [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io), and [Hugging Face](https://huggingface.co/).

## Features
- Upload multiple PDF files
- Ask natural language questions
- Answers pulled from document chunks using RAG
- Powered by Hugging Face LLMs + Sentence Transformers
- Live Streamlit interface

## Tech Stack
- Python
- LangChain
- Streamlit
- Hugging Face Hub (Flan-T5 model)
- Chroma for vector search
- HuggingFaceEmbeddings (MiniLM-L6-v2)

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