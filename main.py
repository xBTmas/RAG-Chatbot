from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

from transformers import pipeline

import os
from dotenv import load_dotenv

import streamlit as st 

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API-KEY")

st.title("Question and answers")

if 'messages' not in st.session_state:
        st.session_state.messages=[]
for message in st.session_state.messages:
     st.chat_message(message['role']).markdown(message['content'])

st.markdown("""
    <style>
        div[role="radiogroup"] {
            text-align: left; /* Aligns the radio buttons to the left */
        }
    </style>
""", unsafe_allow_html=True)

mode = st.radio("Choose an option:", ["Ask a Question", "Summarize Document"])

prompt = st.chat_input("Ask your question here")

uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

summarizer = pipeline("summarization", model="google-t5/t5-small")
if mode == "Summarize Document":
    if uploaded_files:
        all_text = ""
        print(all_text)
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name,"wb") as t:
                t.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(uploaded_file.name)
            pages = loader.load()
            text = " ".join([p.page_content for p in pages])
            chunk_size = 1024
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            summaries = [summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] 
                         for chunk in text_chunks
            ]
            
            full_summary = " ".join(summaries)

            st.subheader(f"Summary of {uploaded_file.name}")
            st.write(full_summary)
    else:
        st.warning("Please upload a document first.")

if mode == "Ask a Question":
    if uploaded_files:
        all_chunks=[]
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name,"wb") as t:
            t.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(uploaded_file.name)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)

        all_chunks.extend(chunks)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={"device":"cpu"}, 
                                           encode_kwargs={"normalize_embeddings": True})

        unique_chunks = list({chunk.page_content for chunk in all_chunks})
        vector_db = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory="chroma_db")
        llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.7, "max_length": 500})

        retriever=vector_db.as_retriever(search_type="similarity",search_kwargs={"k":1})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    response = qa_chain.run(prompt)
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})