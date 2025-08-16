"""
4_embeddings_deep_dive.py - experiment with varous types of embeddings

@author: Manish Bhobé
My experiments with Python, AI/ML and Gen AI.
Code is shared for learning purposed only - use at own risk!
"""

import sys, os, time
from pathlib import Path
from typing import List

append_to_sys_path = Path(__file__).parent.parent
if str(append_to_sys_path) not in sys.path:
    sys.path.append(str(append_to_sys_path))

from dotenv import load_dotenv

from rich.console import Console
from utils.rich_logging import get_logger

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


# load API keys
load_dotenv()
console = Console()
logger = get_logger()

book_path = Path(__file__).parent / "books" / "odyssey.txt"
chromadb_dir = Path(__file__).parent / "chroma_db"
hugfacedb_dir = Path(__file__).parent / "hugging_db"

if not chromadb_dir.exists():
    chromadb_dir.mkdir(parents=True, exist_ok=True)

if not hugfacedb_dir.exists():
    hugfacedb_dir.mkdir(parents=True, exist_ok=True)

# check that we have the paths!
if not book_path.exists():
    raise FileNotFoundError(f"FATAL ERROR: book path '{book_path}' does not exist!")
if not chromadb_dir.exists():
    raise FileNotFoundError(
        f"FATAL ERROR: chromadb base path '{chromadb_dir}' does not exist!"
    )

# load the document into our text loader
loader = TextLoader(str(book_path), encoding="utf-8")
documents = loader.load()

# split into docs
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

console.print(f"[blue]{'='*10} Documents Chunks Info {'='*10}[/blue]")
console.print(f"No of chunks: [yellow]{len(chunks)}[/yellow]")
console.print(f"Sample chunks:\n {chunks[0].page_content[:100]}...")


def create_vector_store(doc_chunks, embeddings, store_name):
    persistent_dir = chromadb_dir / store_name
    if not persistent_dir.exists():
        console.print(f" Creating {str(store_name)}...", end="")
        db = Chroma.from_documents(
            doc_chunks,
            embeddings,
            persist_directory=str(persistent_dir),
        )
        console.print("[yellow]done![/yellow]")
    else:
        console.print(
            f"[green]Vector store {store_name} has already been created![/green]"
        )


console.print(f"[dim green]{'='*10} Google Embeddings {'='*10}[/dim green]")
store_name = "google_embeddings"
persistent_dir = chromadb_dir / store_name
if not persistent_dir.exists():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    create_vector_store(chunks, embeddings, store_name)
else:
    console.print(f"[yellow] vector store {store_name} already exists![/yellow]")

console.print(f"[dim green]{'='*10} HuggingFace Embeddings {'='*10}[/dim green]")
store_name = "huggingface_embeddings"
persistent_dir = chromadb_dir / store_name
if not persistent_dir.exists():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    create_vector_store(chunks, embeddings, store_name)
else:
    console.print(f"[yellow] vector store {store_name} already exists![/yellow]")

sys.exit(0)


def query_vector_store(store_name, query):
    persistent_dir = chromadb_dir / store_name
    if persistent_dir.exists():
        console.print(f"[green]Querying vector store {store_name}[green]")
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma(
            persist_directory=str(persistent_dir),
            embedding_function=embeddings,
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        # display the recovered doc
        console.print("[blue]Relevant docs: [/blue]")
        for i, doc in enumerate(relevant_docs):
            console.print(f"[yellow]Document {i+1}:[/yellow]\n")
            console.print(f"{doc.page_content}\n")
            if doc.metadata:
                console.print(
                    f"[yellow]Source: [/yellow] {doc.metadata.get('source','Unknown')}"
                )
    else:
        console.print(f"[red]Vector store {store_name} does not exist![/red]")


# now let's try a query with different stores
query = "How did Juliet die?"

# Query each vector store
query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_tokens", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)
