"""
chains_parallel.py - run chains in parallel

@Author: Manish Bhobe
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

# load all environment variables
load_dotenv()

# create my LLM - using Google Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# only for colorful text & markdown output support
console = Console()

# messages for PromptTemplate (NOTE: SystemMessage must be first in the list
messages = [
    (
        "system",
        "You are an expert product reviewer who can explain features in a simple language "
        + "and compare features with competing products and give an unbiased comparison.",
    ),
    ("human", "List all the main features and standouts of {product}"),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

# create our runnable lambdas
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# now build a chain
chain = prompt_template | model
# and invoke it
response = chain.invoke({"product": "iPhone 15"})

from rich.markdown import Markdown

console.print(f"[green]Chain output:[/green]\n")
console.print(Markdown(response.content))
