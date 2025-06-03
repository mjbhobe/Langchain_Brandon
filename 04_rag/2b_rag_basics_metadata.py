"""
1b_rag_basics_metadata.py - query the vector store created in 1a_rag_basics_metadata.py
  NOTE: we do not call LLM yet!

@author: Manish Bhobe
My experiments with Python, AI/ML and Gen AI.
Code is shared for learning purposed only - use at own risk!
"""

import sys, os, time
from pathlib import Path

append_to_sys_path = Path(__file__).parent.parent
if str(append_to_sys_path) not in sys.path:
    sys.path.append(str(append_to_sys_path))

from dotenv import load_dotenv

from rich.console import Console
from utils.rich_logging import get_logger

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# load API keys
load_dotenv()
console = Console()
logger = get_logger()

chromadb_index_path = Path(__file__).parent / "chroma_db" / "index_with_metadata"
source_docs_path = Path(__file__).parent / "books"

if not chromadb_index_path.exists():
    # create the embeddings
    console.print(
        "[yellow]Persistent directory does not exist. Please run 1a_rag_basics_metadata.py to create it[/yellow]"
    )
else:
    logger.info(f"Chromadb index already created at {str(chromadb_index_path)}")
    console.print(
        f"[green]Chromadb index already created at {str(chromadb_index_path)}[/green]"
    )

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # and the vector store
    vector_store = Chroma(
        persist_directory=str(chromadb_index_path),
        embedding_function=embeddings,
    )

    # Initialize retriever (retrieve 3 nearest semantically similar chunks)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.1},
    )

    # now we will pass in a query & get relevant docs + metadata from vector store
    query = ""
    while True:
        console.print("[green]Your query? [/green]", end=" ")
        query = input().strip().lower()
        if query in [")quit", "exit", "bye"]:
            break

        retrieved_docs = retriever.invoke(query)
        console.print("[blue]Retrieved docs:[/blue]\n")
        for i, doc in enumerate(retrieved_docs):
            console.print(f"[yellow]Document #{i:2d}[/yellow]")
            console.print(doc.page_content)
            console.print(f"\n[green]Source:[/green]{doc.metadata['source']}\n")
