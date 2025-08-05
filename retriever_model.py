import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatPerplexity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from config import *

def load_all_documents():
    """Loads and splits all PDFs from INPUT_DIR"""
    all_documents = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(INPUT_DIR, filename)
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents) 

    print(f"Loaded {len(all_documents)} total pages")

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    
    chunks = splitter.split_documents(all_documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def build_faiss_index(chunks, save_path="faiss_index_open"):
    """Embeds chunks and builds FAISS index"""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}  # Euclidean distance
    )
    vector_store = FAISS.from_documents(chunks, hf)
    vector_store.save_local(save_path)
    print(f"Vector store saved to {save_path}")
    return vector_store

def load_retriever(index_path="faiss_index_open"):
    """Loads FAISS index and returns retriever"""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    vector_store = FAISS.load_local(index_path, hf, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

def build_qa_chain(retriever):
    """Creates RetrievalQA chain using Perplexity"""
    llm = ChatPerplexity(
        model="sonar",
        pplx_api_key=API_KEY,
        temperature=0.2
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain