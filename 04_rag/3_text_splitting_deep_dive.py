"""
3_text_splitting_deep_dive.py - exploring various text splitting methods in LangChain
    We'll be using Google embeddings & ChromaDb but various techniques to split 1
    text document. We'll choose (randomly) romeo_and_juliet.txt

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
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)


# load API keys
load_dotenv()
console = Console()
logger = get_logger()

book_path = Path(__file__).parent / "books" / "romeo_and_juliet.txt"
chromadb_dir = Path(__file__).parent / "chroma_db"

if not chromadb_dir.exists():
    chromadb_dir.mkdir(parents=True, exist_ok=True)

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

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# function to create vector store
def create_vector_store(chunks, store_name):
    persistent_dir = chromadb_dir / store_name
    if not persistent_dir.exists():
        console.print(f" Creating {store_name}...", end="")
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=str(persistent_dir),
        )
        console.print("[yellow]done![/yellow]")
    else:
        console.print(
            f"[green]Vector store {store_name} has already been created![/green]"
        )


# Character splitting: use for normal RAG applications
console.print("[green]Using character text splitting[/green]", end="")
store_name = "chroma_db_char"
persistent_dir = chromadb_dir / store_name
if not persistent_dir.exists():
    char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    char_chunks = char_splitter.split_documents(documents)
    create_vector_store(char_chunks, store_name)
else:
    console.print(f"[yellow] vector store {store_name} already exists![/yellow]")

# Sentence Splitting: when you want your text split on sentence boundaries, such
# as the usual punctuations (full stop or semi colons for example). This keeps sentences
# together rather than split on arbitrary chunk sizes, which could overlap sentences.
# This is ideal for keeping semantic coherence within chunks
console.print("[green]Using sentence text splitting[/green]", end="")
store_name = "chroma_db_sent"
persistent_dir = chromadb_dir / store_name
if not persistent_dir.exists():
    sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
    sentence_chunks = sentence_splitter.split_documents(documents)
    create_vector_store(sentence_chunks, store_name)
else:
    console.print(f"[yellow] vector store {store_name} already exists![/yellow]")

# split by tokens (typically words or subwords) - this is likely to download
# a tokenizer, such as GPT-2
# Useful for transformer models with strict token limits
console.print("[green]Using token text splitting[/green]", end="")
store_name = "chroma_db_tokens"
persistent_dir = chromadb_dir / store_name
if not persistent_dir.exists():
    token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
    token_chunks = token_splitter.split_documents(documents)
    create_vector_store(token_chunks, store_name)
else:
    console.print(f"[yellow] vector store {store_name} already exists![/yellow]")


# Recursive character splitting (most popular!)
# attempts to split document at logical boundaries (such as sentences, paras) within
# character limits. Balance between Sentence splitter & token splitter
console.print("[green]Using recursive char splitting[/green]", end="")
store_name = "chroma_db_rec_char"
persistent_dir = chromadb_dir / store_name
if not persistent_dir.exists():
    rec_char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    rec_char_chunks = rec_char_splitter.split_documents(documents)
    create_vector_store(rec_char_chunks, store_name)
else:
    console.print(f"[yellow] vector store {store_name} already exists![/yellow]")


# Custom splitter - use your logic. Anything goes
# In this example, we are splitting on 2 consequitive new lines
class CustomTextSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        return text.split("\n\n")


console.print("[green]Using custom text splitting[/green]", end="")
store_name = "chroma_db_custom"
persistent_dir = chromadb_dir / store_name
if not persistent_dir.exists():
    custom_splitter = CustomTextSplitter()
    custom_chunks = custom_splitter.split_documents(documents)
    create_vector_store(custom_chunks, store_name)
else:
    console.print(f"[yellow] vector store {store_name} already exists![/yellow]")


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
