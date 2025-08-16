"""
1b_rag_basics.py - retrieve semantically similar documents from vector store
  NOTE: we do not call LLM yet!

@author: Manish Bhobé
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

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# load API keys
load_dotenv()
console = Console()
logger = get_logger()

faiss_index_path = Path(__file__).parent / "faiss_index"

if not faiss_index_path.exists():
    # create the embeddings
    logger.fatal(
        f"FATAL ERROR: vector store not found! Expected at {str(faiss_index_path)}"
    )
    console.print(
        f"[red]FATAL ERROR: vector store not found! Expected at {str(faiss_index_path)}[/red]"
    )
    sys.exit(-1)
else:
    logger.info(f"Loading FAISS index from {str(faiss_index_path)}")

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the FAISS vector store
    vector_store = FAISS.load_local(
        str(faiss_index_path), embeddings, allow_dangerous_deserialization=True
    )

    # Initialize retriever (retrieve 3 nearest semantically similar chunks)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.4},
    )

    # Initialize LLM
    # create my LLM - using Google Gemini
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash",
    #     temperature=0.2,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     # other params...
    # )

    # # Prompt template
    # template = """
    # You are an AI assistant helping with questions based on the following context:
    # {context}

    # Question: {question}
    # Answer as best as possible based on the context above.
    # """

    # prompt = PromptTemplate.from_template(template)
    # chain = (
    #     {"context": retriever, "question": lambda x: x["question"]}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # # Create RetrievalQA chain
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True,
    # )

    # infinite loop
    question = ""
    while True:
        console.print("[green]Your question? [/green]")
        question = input().lower().strip()
        if question in ["exit", "quit", "bye"]:
            console.print("[red]Exiting...[/red]")
            break

        # ask the vector store to retrieve semantically similar docs
        relevant_docs = retriever.invoke(question)
        console.print("[yellow]Relevant documents:[/yellow]\n")
        for doc in relevant_docs:
            console.print(f"[blue]{'-*-'*20}[/blue]")
            console.print(Markdown(doc.page_content))
            console.print("\n")

        # run your chain
        # logger.debug(f"Asking LLM to respond to {question}")
        # #result = qa_chain.invoke({"query": question})
        # console.print("[yellow]Answer:[/yellow]\n")
        # console.print(Markdown(result["result"]))
