"""Local Streamlit chat for the cancer-awareness document prototype."""

import html
import re

import streamlit as st

from rag import TECHNICAL_MESSAGE, answer_question, transcribe_speech
from supabase_store import (
    NOT_CONFIGURED,
    SupabaseStoreError,
    clear_chat_messages,
    configured,
    load_chat_messages,
    load_hospital_profile,
    save_chat_message,
)

st.set_page_config(page_title="Cancer Awareness Assistant")
st.title("Cancer Awareness Assistant")
st.caption(
    "Educational information from synthetic sample documents for this engineering prototype. "
    "These samples are not hospital-approved clinical content. "
    "This tool does not diagnose or recommend treatment."
)
st.caption(
    "Urgent symptoms get a fixed safety message. "
    "Other questions are answered from the hospital profile keys and the loaded documents first. "
    "The web is used only when those sources do not contain the answer."
)


def render_hospital(profile: dict | None) -> None:
    if not profile:
        return
    fonts = profile.get("fonts") or ["Arial", "Helvetica"]
    font_family = ", ".join(html.escape(str(font)) for font in fonts)
    swatches = "".join(
        (
            '<span style="display:inline-block;width:0.9rem;height:0.9rem;'
            "margin-right:0.35rem;vertical-align:middle;border:1px solid #ccc;"
            f'background:{color};"></span>'
        )
        for color in profile.get("colors") or []
        if re.fullmatch(r"#[0-9A-Fa-f]{3,8}", str(color))
    )
    name = html.escape(str(profile.get("name", "")))
    website = html.escape(str(profile.get("website", "")), quote=True)
    st.markdown(
        (
            f'<p style="font-family:{font_family},sans-serif;margin-bottom:0.15rem;">'
            f"<strong>{name}</strong></p>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f'{profile.get("address", "")} · {profile.get("phone", "")} · {profile.get("email", "")}'
    )
    st.markdown(
        f'{swatches} <a href="{website}">{website}</a>',
        unsafe_allow_html=True,
    )
    st.caption(f'Logo file: {profile.get("logo_url", "")}')


def persist_message(role: str, content: str) -> None:
    if not st.session_state.get("supabase_ready"):
        return
    try:
        save_chat_message(role, content)
    except SupabaseStoreError as exc:
        st.warning(str(exc))


if "supabase_ready" not in st.session_state:
    st.session_state.hospital = None
    st.session_state.supabase_notice = None
    if configured():
        try:
            st.session_state.hospital = load_hospital_profile()
            st.session_state.messages = load_chat_messages()
            st.session_state.supabase_ready = True
        except SupabaseStoreError as exc:
            st.session_state.messages = []
            st.session_state.supabase_notice = str(exc)
            st.session_state.supabase_ready = False
    else:
        st.session_state.messages = []
        st.session_state.supabase_notice = NOT_CONFIGURED
        st.session_state.supabase_ready = False

render_hospital(st.session_state.hospital)
if st.session_state.supabase_notice:
    st.warning(st.session_state.supabase_notice)
elif st.session_state.supabase_ready:
    st.caption(
        "The hospital profile, indexed documents, and this chat are stored in Supabase. "
        "Do not enter personal health information."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear chat", disabled=not st.session_state.messages):
    if st.session_state.get("supabase_ready"):
        try:
            clear_chat_messages()
        except SupabaseStoreError:
            st.session_state.clear_error = (
                "The chat on screen was cleared, but Supabase did not delete the saved messages. "
                "Run sql/schema.sql again so the delete policy is in place."
            )
    st.session_state.messages = []
    st.session_state.clear_notice = "Chat history cleared."
    st.rerun()

if notice := st.session_state.pop("clear_notice", None):
    st.success(notice)
if error := st.session_state.pop("clear_error", None):
    st.warning(error)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def ask(question: str) -> None:
    question = question.strip()
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    persist_message("user", question)
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the hospital profile and documents first..."):
            try:
                answer = answer_question(question)
            except Exception:
                answer = TECHNICAL_MESSAGE
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    persist_message("assistant", answer)


audio = st.audio_input("Speak a question")
if audio is not None:
    signature = (getattr(audio, "file_id", None), audio.size)
    if signature != st.session_state.get("voice_signature"):
        st.session_state.voice_signature = signature
        with st.spinner("Transcribing your question..."):
            try:
                spoken = transcribe_speech(audio.getvalue(), audio.name or "speech.wav")
            except Exception:
                spoken = None
        if spoken:
            st.session_state.voice_error = None
            ask(spoken)
        elif spoken == "":
            st.session_state.voice_error = (
                "No speech was detected. Try again or type the question."
            )
        else:
            st.session_state.voice_error = (
                "The recording could not be transcribed. Try again or type the question."
            )

if voice_error := st.session_state.get("voice_error"):
    st.warning(voice_error)

question = st.chat_input("Ask an educational question")
if question:
    st.session_state.voice_error = None
    ask(question)
