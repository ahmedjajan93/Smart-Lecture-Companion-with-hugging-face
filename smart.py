# smart_lecture_companion/app.py

import streamlit as st
from PIL import Image
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFaceHub
import tempfile
from langchain_community.embeddings import OllamaEmbeddings
import os
import torch
from langchain.llms import Ollama
import re


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="Smart Lecture Companion", layout="wide")
st.title("📚 Smart Lecture Companion")

uploaded_file = st.file_uploader("Upload Lecture Notes or Slides (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("File uploaded successfully!")

    # Load and embed the document
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    embeddings = OllamaEmbeddings(model='nomic-embed-text') 
    db = FAISS.from_documents(documents, embeddings)

    # Set up QA system
   # Initialize Ollama (cached)
    llm = Ollama(model="deepseek-r1:1.5b")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    st.header("Ask Questions About Your Lecture 📖")
    query = st.text_input("What would you like to know?")

    if query:
        answer = qa.run(query)
        cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
       
        st.write("**Answer:**", cleaned)

    st.header("✨ Key Points Summary")
    if st.button("Generate Summary"):
        summary_prompt = "Summarize the most important points from this lecture."
        summary = qa.run(summary_prompt)
        st.success(summary_prompt)

    st.header("🧠 Auto-Generated Flashcards")
    if st.button("Create Flashcards"):
        flashcard_prompt = "Generate 5 flashcards from this lecture with questions and answers."
        flashcards = qa.run(flashcard_prompt)
        st.info(flashcards)

    os.remove(tmp_path)
else:
    st.info("Upload a PDF to get started.")
