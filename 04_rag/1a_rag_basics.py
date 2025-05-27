"""
1a_rag_basics.py - setting up embeddings for RAG using Google Gemini
    vector store & embeddings

@author: Manish Bhobe
My experiments with Python, AI/ML and Gen AI.
Code is shared for learning purposed only - use at own risk!
"""

import pickle
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from utils.rich_logging import get_logger

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# load API keys
load_dotenv()
console = Console()
logger = get_logger()

faiss_index_path = Path(__file__).parent / "faiss_index"

if not faiss_index_path.exists():
    # create the embeddings
    logger.info("Creating offline vector store")

    # build path to the file we want to embed
    source_docs_path = Path(__file__).parent / "books" / "odyssey.txt"
    if not source_docs_path.exists():
        logger.fatal(f"FATAL ERROR: could not find path {source_docs_path}")
    assert (
        source_docs_path.exists()
    ), f"FATAL ERROR: could not find path {source_docs_path}"

    # load data from source file
    data_loader = TextLoader(str(source_docs_path), encoding="utf-8")
    documents = data_loader.load()
    logger.info(f"[green]Documents:[/green]{documents}")

    # split documents into manageable chunks
    console.print(f"[green]Splitting documents...[/green]")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"[yellow]No of chunks:[/yellow]{len(chunks)}")

    # build vector store & save offline
    console.print(f"[green]Creating embeddings...[/green]")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(faiss_index_path))
    logger.info(f"Embeddings saved to path {str(faiss_index_path)}")
else:
    logger.info(f"FAISS index already created at {str(faiss_index_path)}")
