import os
import tempfile
import streamlit as st
from streamlit_chat import message
from ingest_rag import ChatFromYourData
from typing import List, Tuple

st.set_page_config(page_title="ChatFromYourData")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "assistant" not in st.session_state:
    st.session_state["assistant"] = ChatFromYourData()
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "ingestion_complete" not in st.session_state:
    st.session_state["ingestion_complete"] = False

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
    # commented: st.session_state["assistant"].clear()
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