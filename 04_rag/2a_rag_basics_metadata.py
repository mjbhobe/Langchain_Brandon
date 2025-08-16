"""
1a_rag_basics_metadata.py - build the vector store in the chroma_index_with_metadata
    directory that holds embeddings & meta data for multiple books
  NOTE: we do not call LLM yet!

@author: Manish Bhobé
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
from rich.progress import Progress, BarColumn, TextColumn
from rich.markdown import Markdown
from utils.rich_logging import get_logger

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
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
        "[yellow]Persistent directory does not exist. Initializing vector store...[/yellow]"
    )

    # build path to the file we want to embed

    if not source_docs_path.exists():
        logger.fatal(f"FATAL ERROR: could not find path {source_docs_path}")
        console.print(f"[red]FATAL ERROR: could not find path {source_docs_path}[/red]")
        raise FileNotFoundError(
            f"[red]FATAL ERROR: could not find path {source_docs_path}[/red]"
        )

    # load all .txt files from the books folder
    book_files = [f for f in os.listdir(source_docs_path) if f.endswith("txt")]
    num_files = len(book_files)
    logger.debug(f"Files detected: {book_files}")

    bar_width = 40
    # for book in book_files:
    with Progress(
        TextColumn("Processing files: ", style="white"),
        BarColumn(
            bar_width=bar_width,
            complete_style="green",
            finished_style="green",
            pulse_style="green",
            style="white",
        ),
        TextColumn("[bold yellow]{task.fields[filename]}", justify="left"),
        transient=True,
    ) as progress:
        task = progress.add_task("process", total=num_files, filename=" " * 20)
        for book in book_files:
            progress.update(task, advance=1, filename=book)
            documents = []
            book_path = os.path.join(source_docs_path, book)
            # progress.update(task, description=f"[green]Processing: {book}[/green]")
            loader = TextLoader(book_path, encoding="utf-8")
            book_docs = loader.load()
            # set source metadata for each book_doc
            for book_doc in book_docs:
                book_doc.metadata = {"source": book_path}
                documents.append(book_doc)

            # split into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=25)
            chunks = text_splitter.split_documents(documents)

            # embed
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = Chroma.from_documents(
                chunks,
                embeddings,
                persist_directory=str(chromadb_index_path),
            )

            # print(documents)

        logger.info(f"Embeddings saved to path {str(chromadb_index_path)}")
        console.print(
            f"[green]Embeddings saved to path[/green] {str(chromadb_index_path)}"
        )
else:
    logger.info(f"Chromadb index already created at {str(chromadb_index_path)}")
    console.print(
        f"[green]Chromadb index already created at {str(chromadb_index_path)}[/green]"
    )

    # Initialize embeddings
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # # Load the FAISS vector store
    # vector_store = FAISS.load_local(
    #     str(chromadb_index_path), embeddings, allow_dangerous_deserialization=True
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
