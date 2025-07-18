"""
1a_rag_basics.py - build the vector store in the faiss_index directory
  NOTE: we do not call LLM yet!

@author: Manish Bhobe
My experiments with Python, AI/ML and Gen AI.
Code is shared for learning purposed only - use at own risk!
"""

import sys
from pathlib import Path

append_to_sys_path = Path(__file__).parent.parent
if str(append_to_sys_path) not in sys.path:
    sys.path.append(str(append_to_sys_path))

from dotenv import load_dotenv

from rich.console import Console
from rich.markdown import Markdown
from utils.rich_logging import get_logger

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA

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
    logger.info(f"FAISS index created at {str(faiss_index_path)}")
    console.print(f"[green]FAISS index created at {str(faiss_index_path)}[/green]")

    # Initialize embeddings
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # # Load the FAISS vector store
    # vector_store = FAISS.load_local(
    #     str(faiss_index_path), embeddings, allow_dangerous_deserialization=True
    # )

    # # Initialize retriever (retrieve 3 nearest semantically similar chunks)
    # retriever = vector_store.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"k": 3, "score_threshold": 0.4},
    # )

    # # Initialize LLM
    # # create my LLM - using Google Gemini
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash",
    #     temperature=0.2,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     # other params...
    # )

    # # # Prompt template
    # # template = """
    # # You are an AI assistant helping with questions based on the following context:
    # # {context}

    # # Question: {question}
    # # Answer as best as possible based on the context above.
    # # """

    # # prompt = PromptTemplate.from_template(template)
    # # chain = (
    # #     {"context": retriever, "question": lambda x: x["question"]}
    # #     | prompt
    # #     | llm
    # #     | StrOutputParser()
    # # )

    # # Create RetrievalQA chain
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True,
    # )

    # # infinite loop
    # question = ""
    # while True:
    #     console.print("[green]Your question? [/green]")
    #     question = input().lower().strip()
    #     if question in ["exit", "quit", "bye"]:
    #         logger.debug(f"You entered {question} - exiting!")
    #         console.print("[red]Exiting...[/red]")
    #         break

    #     # run your chain
    #     logger.debug(f"Asking LLM to respond to {question}")
    #     result = qa_chain.invoke({"query": question})
    #     console.print("[yellow]Answer:[/yellow]\n")
    #     console.print(Markdown(result["result"]))
