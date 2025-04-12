import streamlit as st
from PIL import Image
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
import os
import torch
import re





# Set up Hugging Face API token securely (using Streamlit Secrets Manager)
hf_token = os.getenv('OPEAN_API')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="Smart Lecture Companion", layout="wide")
st.title("📚 Smart Lecture Companion")

# File upload
uploaded_file = st.file_uploader("Upload Lecture Notes or Slides (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("File uploaded successfully!")

    # Load and embed the document
    loader = PyMuPDFLoader(tmp_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    # Model loading with error handling
    llm = ChatOpenAI(
        model="mistralai/mistral-small-3.1-24b-instruct:free",  # or any model you want from OpenRouter
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=hf_token 
)

 

    # Set up QA system
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    st.header("Ask Questions About Your Lecture 📖")
    query = st.text_input("What would you like to know?")

    if query:
        answer = qa.run(query)
        cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        st.write("**Answer:**", cleaned)

    # Key Points Summary
    st.header("✨ Key Points Summary")
    if st.button("Generate Summary"):
        summary_prompt = "Summarize the most important points from this lecture."
        summary = qa.run(summary_prompt)
        st.success(summary)

    # Auto-generated Flashcards
    st.header("🧠 Auto-Generated Flashcards")
    if st.button("Create Flashcards"):
        flashcard_prompt = "Generate 5 flashcards from this lecture with questions and answers."
        flashcards = qa.run(flashcard_prompt)
        st.info(flashcards)

    os.remove(tmp_path)
else:
    st.info("Upload a PDF to get started.")
