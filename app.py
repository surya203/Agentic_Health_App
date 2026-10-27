"""Local Streamlit chat for the cancer-awareness document prototype."""

import streamlit as st

from rag import TECHNICAL_MESSAGE, answer_question

st.set_page_config(page_title="Cancer Awareness Assistant")
st.title("Cancer Awareness Assistant")
st.caption(
    "Educational information from synthetic sample documents for this engineering prototype. "
    "These samples are not hospital-approved clinical content. "
    "This tool does not diagnose or recommend treatment."
)
st.caption(
    "Urgent symptoms are answered with a fixed safety message. "
    "A question is sent to Groq, with the matching passages, only after local safety and evidence checks pass."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask an educational question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching loaded documents..."):
            try:
                answer = answer_question(question)
            except Exception:
                answer = TECHNICAL_MESSAGE
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
