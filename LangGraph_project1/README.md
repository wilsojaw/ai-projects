# 📱 ChatBot Genie — Built with LangGraph & Streamlit

This is a phone-style chatbot powered by [LangGraph](https://github.com/langchain-ai/langgraph) and OpenAI. You get **three questions** — like a genie granting wishes — followed by a conversation summary.

### 🧠 Features
- LangGraph event-driven LLM flow
- Chat-style UI in Streamlit
- Follows up naturally after every user input
- Auto-generates a summary after the last turn

---

### 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run graph_app.py
```

> Requires an `.env` file with your OpenAI key:
```env
OPENAI_API_KEY=your-key-here
```

---

### 🌐 Run on Hugging Face Spaces

To deploy:
1. Push this repo to Hugging Face Spaces
2. Choose **"Streamlit"** as the SDK
3. Add your API key as a **Secret** (OPENAI_API_KEY)

Enjoy your 3-wish genie chatbot ✨
