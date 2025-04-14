import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

class GraphState(TypedDict):
    input: str
    bot_response: str
    turn: int
    history: list[str]
    final: bool

def llm_query_node(state: GraphState) -> GraphState:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    history_text = "\n".join(state["history"])
    if state.get("final", False):
        prompt = f"""
You are an outrageously snarky genie. Briefly acknowledge the user's final wish with a witty, sarcastic remark. For example, you might say 'Last one, you better make it count' or 'You know you're getting rated on these, right?'
Conversation so far:
{history_text}
User: {state['input']}
Genie:"""
    else:
        prompt = f"""
You are an outrageously snarky genie. Briefly acknowledge the user's wish with a witty, sarcastic remark. Avoid overly positive language and keep it short. For instance, you could say 'Another wish, huh? Really?' or 'Well, that's one way to do it.'
Conversation so far:
{history_text}
User: {state['input']}
Genie:"""
    response = llm.invoke(prompt)
    return {
        **state,
        "bot_response": response.content.strip()
    }

builder = StateGraph(GraphState)
builder.add_node("llm_query", RunnableLambda(llm_query_node))
builder.set_entry_point("llm_query")

builder.add_edge("llm_query", END)
graph = builder.compile()

st.set_page_config(page_title="🧞 Genie Chat Bot", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧞 Genie Chat Bot 😤</h1>", unsafe_allow_html=True)

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
    line-height: 1.4;
}
.user-bubble {
    background-color: #4CAF50;
    color: white;
    align-self: flex-end;
}
.bot-bubble {
    background-color: #2A2A2A;
    color: #f1f1f1;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

if "state" not in st.session_state:
    st.session_state.state: GraphState = {
        "input": "",
        "bot_response": "",
        "turn": 0,
        "history": [],
        "final": False
    }

if "valid_wishes" not in st.session_state:
    st.session_state.valid_wishes = 0

if "user_wishes" not in st.session_state:
    st.session_state.user_wishes = []

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.state["history"]:
    bubble_class = "user-bubble" if msg.startswith("User:") else "bot-bubble"
    text = msg.replace("User: ", "").replace("Bot: ", "")
    st.markdown(f"<div class='{bubble_class}'>{text}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat logic
if st.session_state.valid_wishes < 3:
    if st.session_state.valid_wishes == 0:
        prompt_text = "What is your wish?"
    elif st.session_state.valid_wishes == 2:
        prompt_text = "This is your final wish"
    else:
        prompt_text = "What is your next wish?"
    user_input = st.chat_input(prompt_text)
    if user_input:
        # Record every user wish regardless of content
        st.session_state.user_wishes.append(user_input)
        forbidden_keywords = ["more wishes", "unlimited wishes", "wish for more wishes", "love", "fall in love", "make someone love me", "resurrect", "bring back", "raise the dead"]
        if any(bad in user_input.lower() for bad in forbidden_keywords):
            genie_warning = "🧞‍♂️ Ahh, but I cannot grant that! The rules are clear: no love, no resurrection, and no wishing for more wishes. You have forfeited this wish as payment for that offense."
            st.session_state.state["history"].append(f"User: {user_input}")
            st.session_state.state["history"].append(f"Bot: {genie_warning}")
        else:
            if st.session_state.valid_wishes == 2:
                st.session_state.state["final"] = True
            else:
                st.session_state.state["final"] = False
            st.session_state.state["input"] = user_input
            st.session_state.state = graph.invoke(st.session_state.state)
            st.session_state.state["history"].append(f"User: {user_input}")
            st.session_state.state["history"].append(f"Bot: {st.session_state.state['bot_response']}")
        st.session_state.state["turn"] += 1
        st.session_state.valid_wishes += 1
        st.rerun()
else:
    # All three wishes have been submitted; now rate each wish individually
    st.markdown("<h3>🌟 Genie’s Ratings:</h3>", unsafe_allow_html=True)
    # Higher temperature doesn't increase API usage count, but it makes the genie snarkier!
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, presence_penalty=0.5, frequency_penalty=0.2)
    for i, wish in enumerate(st.session_state.user_wishes, start=1):
        rating_prompt = (
            f"You are an outrageously snarky, sarcastic, and witty genie. "
            f"Critically evaluate the following wish on a scale from 1 to 10, ensuring your numerical rating is accurate. "
            f"If the wish is dull or unimpressive, the rating should be low (typically below 5). "
            f"However, if the wish demonstrates creativity and kindness—such as caring for others or animals—it should receive a high score (typically above 5). "
            f"Accompany your rating with a snarky, witty explanation that justifies the score. "
            f"Remember: higher temperature doesn't cost more, it just makes your snark even sharper!\nWish {i}: {wish}"
        )
        rating = llm.invoke(rating_prompt).content.strip()
        st.markdown(f"<div class='bot-bubble'><strong>Wish {i}:</strong> {wish}<br/><em>Rating: {rating}</em></div>", unsafe_allow_html=True)
    st.text_input("Wishes fulfilled.", value="", disabled=True)
