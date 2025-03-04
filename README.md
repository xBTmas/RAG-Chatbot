PDF Q&A and Summarization App

Overview

This project is a PDF-based Question Answering (Q&A) and Summarization web application built using LangChain, Hugging Face models, and Streamlit. It allows users to upload PDF documents, ask questions about their content, and generate concise summaries. The application is containerized with Docker for easy deployment and scalability.

Features

Q&A System – Users can ask questions based on uploaded PDFs, powered by Retrieval-Augmented Generation (RAG).

Document Summarization – Generates concise summaries using Google T5-small.

Vector Search with ChromaDB – Embeddings are generated using Hugging Face's Sentence Transformers.

LLM-powered Response Generation – Uses Falcon-7B for intelligent answers.

Streamlit-based UI – Interactive and user-friendly web interface.

Dockerized for Portability – Easily deployable using Docker.

Tech Stack

Frontend & UI: Streamlit

Backend & Processing: LangChain, PyPDFLoader

Machine Learning & NLP: Hugging Face, Falcon-7B, Sentence Transformers

Vector Database: ChromaDB

Deployment: Docker

Installation & Setup

Prerequisites:

Python 3.8+

Docker (if running in a container)

Hugging Face API Key (stored in .env file)

Steps to Run Locally:

Clone the repository:

git clone https://github.com/your-repo/pdf-qna-summarization.git
cd pdf-qna-summarization

Create and activate a virtual environment:

python -m venv env
source env/bin/activate   # For macOS/Linux
env\Scripts\activate      # For Windows

Install dependencies:

pip install -r requirements.txt

Set up environment variables:

Create a .env file and add your Hugging Face API key:

HUGGINGFACEHUB_API_TOKEN=your_api_key_here

Run the application:

streamlit run main.py

Run with Docker:

Build the Docker image:

docker build -t pdf-qna-app .

Run the container:

docker run -p 8501:8501 pdf-qna-app

Usage

Upload a PDF document using the file uploader.

Choose an option:

"Ask a Question" – Enter a question related to the document.

"Summarize Document" – Automatically generate a summary.

View the AI-generated responses displayed in the chat UI.

Future Enhancements

Support for multiple document processing.

Enhanced metadata filtering for refined searches.

Integration with GPU acceleration for faster embeddings and inference.
