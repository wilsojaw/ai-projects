import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

class GraphState(TypedDict):
    input: str
    summary: str
    bot_response: str
    turn: int
    history: list[str]

def summarize_node(state: GraphState) -> GraphState:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    prompt = f"Summarize the following input in 1-2 sentences:\n\n{state['input']}"
    response = llm.invoke(prompt)
    return {
        **state,
        "summary": response.content.strip()
    }

def llm_query_node(state: GraphState) -> GraphState:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    history_text = "\n".join(state["history"])
    prompt = f"""
    {history_text}
    User: {state['input']}
    Bot: Answer naturally, then follow up with a related question.
    """
    response = llm.invoke(prompt)
    return {
        **state,
        "bot_response": response.content.strip()
    }

builder = StateGraph(GraphState)
builder.add_node("summarize", RunnableLambda(summarize_node))
builder.add_node("llm_query", RunnableLambda(llm_query_node))
builder.set_entry_point("summarize")
builder.add_edge("summarize", "llm_query")
builder.add_edge("llm_query", END)
graph = builder.compile()

st.set_page_config(page_title="📱 ChatBot", layout="centered")
st.markdown("<h1 style='text-align: center;'>📱 ChatBot</h1>", unsafe_allow_html=True)

# Custom CSS for phone-style bubbles
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
}
.user-bubble, .bot-bubble {
    max-width: 80%;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 10px;
    font-size: 16px;
}
.user-bubble {
    background-color: #DCF8C6;
    align-self: flex-end;
}
.bot-bubble {
    background-color: #ECECEC;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

if "state" not in st.session_state:
    st.session_state.state: GraphState = {
        "input": "",
        "summary": "",
        "bot_response": "",
        "turn": 0,
        "history": []
    }

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for i, message in enumerate(st.session_state.state["history"]):
    bubble_class = "user-bubble" if message.startswith("User:") else "bot-bubble"
    text = message.replace("User: ", "").replace("Bot: ", "")
    st.markdown(f"<div class='{bubble_class}'>{text}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if st.session_state.state["turn"] < 3:
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.state["input"] = user_input
        st.session_state.state = graph.invoke(st.session_state.state)
        st.session_state.state["history"].append(f"User: {user_input}")
        st.session_state.state["history"].append(f"Bot: {st.session_state.state['bot_response']}")
        st.session_state.state["turn"] += 1
        st.rerun()
elif st.session_state.state["turn"] == 3:
    final_input = st.chat_input("You get one final message...")
    if final_input:
        st.session_state.state["history"].append(f"User: {final_input}")
        st.session_state.state["turn"] += 1
        st.rerun()
else:
    st.markdown("<div class='bot-bubble'>I'm like a genie with only 3 wishes. Goodbye</div>", unsafe_allow_html=True)
    # Generate summary
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    history_text = "\n".join(st.session_state.state["history"])
    summary_prompt = f"Summarize this conversation in 2-3 sentences:\n{history_text}"
    summary_response = llm.invoke(summary_prompt)
    st.markdown("<h3>📋 Conversation Summary:</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-bubble'>{summary_response.content.strip()}</div>", unsafe_allow_html=True)
    st.text_input("Chat ended.", value="", disabled=True)
