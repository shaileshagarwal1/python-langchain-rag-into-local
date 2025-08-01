import os
import tempfile
from typing import List, Optional, Set
from langchain_community.vectorstores.chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document

# --- Imports for Image Handling ---
import base64
from PIL import Image
from io import BytesIO
import hashlib


# Constants for the RAG chain
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
SEARCH_KWARGS = {
    'k': 6,
    'score_threshold': 0.3
}
MODEL_NAME = "qwen3:1.7b"
#MULTIMODAL_MODEL_NAME = "llava" # Recommended multimodal model
PROMPT_TEMPLATE = """
[INST]<<SYS>> You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If the context is empty or does not contain the answer, you should state that the information is not available in the provided documents.
If you have general knowledge about the topic, you may provide a brief, high-level summary, but prioritize information from the context if it exists.
Use mininum four to six sentences maximum and keep the answer concise.<</SYS>>
Question: {question}
Context: {context}
Answer: [/INST]
"""

class ChatFromYourData:
    vector_store: Optional[Chroma] = None
    retriever: Optional[any] = None
    chain: Optional[any] = None
    ingested_files: Set[str] = set()

    def __init__(self):
        self.model = ChatOllama(model=MODEL_NAME)
        #self.multimodal_model = ChatOllama(model=MULTIMODAL_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def _get_llm_chain(self):
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def _update_vector_store(self, chunks: List[Document]):
        if not self.vector_store:
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        else:
            self.vector_store.add_documents(documents=chunks)

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=SEARCH_KWARGS,
        )
        self.chain = self._get_llm_chain()
    
    def _get_file_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def ingest_pdf(self, pdf_path: str):
        file_hash = self._get_file_hash(pdf_path)
        if file_hash in self.ingested_files:
            print("PDF already ingested, skipping.")
            return True
            
        try:
            docs = PyPDFLoader(file_path=pdf_path).load()
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            self._update_vector_store(chunks)
            self.ingested_files.add(file_hash)
            return True
        except Exception as e:
            print(f"An error occurred during PDF ingestion: {e}")
            self.clear()
            raise

    def ingest_image(self, image_path: str):
        file_hash = self._get_file_hash(image_path)
        if file_hash in self.ingested_files:
            print("Image already ingested, skipping.")
            return True

        try:
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.save(buffered, format=img.format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            prompt = "Provide a detailed description of the image content."
            response = self.multimodal_model.invoke([prompt, img_str])
            
            image_doc = Document(page_content=response.content, metadata={"source": image_path, "type": "image"})
            
            chunks = self.text_splitter.split_documents([image_doc])
            self._update_vector_store(chunks)
            self.ingested_files.add(file_hash)
            return True
        except Exception as e:
            print(f"An error occurred during image ingestion: {e}")
            self.clear()
            raise

    def ask(self, query: str) -> str:
        if not self.chain:
            return "Please ingest a PDF or image file first."
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.ingested_files = set()