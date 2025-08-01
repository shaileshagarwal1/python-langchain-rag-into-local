import os
import tempfile
import streamlit as st
from streamlit_chat import message
from ingest_rag import ChatFromYourData
from typing import List, Tuple

st.set_page_config(page_title="ChatFromYourData")

# --- Initialize session state with default values ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "assistant" not in st.session_state:
    st.session_state["assistant"] = None
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "ingestion_complete" not in st.session_state:
    st.session_state["ingestion_complete"] = False
if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = "openai" # Default to OpenAI
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "llm_configured" not in st.session_state:
    st.session_state["llm_configured"] = False


def configure_llm_callback():
    """
    Re-initializes the ChatFromYourData assistant when the LLM provider
    or API key is changed.
    """
    api_key_valid = True
    provider = st.session_state["llm_provider"]
    api_key_required_providers = ["openai", "gemini", "groq3"]

    # Use .get() to safely check for the API key
    if provider in api_key_required_providers and not st.session_state.get("api_key"):
        api_key_valid = False
        st.error(f"Please enter a valid API key for {provider.upper()}.")
    
    if api_key_valid:
        try:
            st.session_state["assistant"] = ChatFromYourData(
                llm_provider=provider,
                # Use .get() here as well for safety
                api_key=st.session_state.get("api_key") if provider in api_key_required_providers else None
            )
            st.session_state["llm_configured"] = True
            st.success(f"LLM provider switched to {provider.upper()}!")
        except ValueError as e:
            st.error(f"Configuration Error: {e}")
            st.session_state["llm_configured"] = False


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if not st.session_state["ingestion_complete"]:
        st.error("Please upload and ingest a document or image first.")
        return

    user_input = st.session_state.get("user_input", "")
    if user_input and len(user_input.strip()) > 0:
        user_text = user_input.strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))
        st.session_state["user_input"] = ""


def read_and_save_file():
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state["ingestion_complete"] = False
    
    file_uploader_key = "file_uploader"
    if file_uploader_key in st.session_state and st.session_state[file_uploader_key] is not None:
        
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting documents..."):
            all_files_ingested = True
            for file in st.session_state["file_uploader"]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.type.split('/')[-1]}") as tf:
                    tf.write(file.getbuffer())
                    file_path = tf.name
                
                try:
                    if file.type == "application/pdf":
                        st.session_state["assistant"].ingest_pdf(file_path)
                    elif file.type in ["image/jpeg", "image/png"]:
                        st.session_state["assistant"].ingest_image(file_path)
                    else:
                        st.warning(f"Unsupported file type: {file.name}")
                        all_files_ingested = False
                except Exception as e:
                    st.error(f"Failed to ingest {file.name}: {e}")
                    all_files_ingested = False
                finally:
                    os.remove(file_path)
            
            if all_files_ingested:
                st.session_state["ingestion_complete"] = True
                st.success("All documents ingested successfully!")
            else:
                st.error("Document ingestion failed.")


def page():
    # --- LLM Provider Selection UI ---
    st.sidebar.header("LLM Configuration")
    provider_options = ("openai", "ollama", "gemini", "groq3")
    st.sidebar.radio(
        "Choose LLM Provider:",
        provider_options,
        key="llm_provider",
        on_change=configure_llm_callback
    )

    provider = st.session_state["llm_provider"]
    api_key_required = provider in ["openai", "gemini", "groq3"]

    if api_key_required:
        st.sidebar.text_input(
            f"{provider.upper()} API Key:",
            type="password",
            key="api_key",
            on_change=configure_llm_callback
        )
        if provider == "openai":
            st.sidebar.markdown("Don't have a key? [Get one here](https://platform.openai.com/api-keys)")
        elif provider == "gemini":
            st.sidebar.markdown("Don't have a key? [Get one here](https://aistudio.google.com/app/apikey)")
        elif provider == "groq3":
            st.sidebar.markdown("Don't have a key? [Get one here](https://console.groq.com/keys)")

    if st.session_state["assistant"] is None or not st.session_state["llm_configured"]:
        configure_llm_callback()

    st.header("Chat From PDF & Images")
    st.subheader("Upload documents or images")
    st.file_uploader(
        "Upload document or image",
        type=["pdf", "jpg", "jpeg", "png"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input, disabled=not st.session_state["ingestion_complete"])


if __name__ == "__main__":
    page()
