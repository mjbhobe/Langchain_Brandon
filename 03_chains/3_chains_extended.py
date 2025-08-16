"""
chains_extended.py - using RunnableLambdas to extend chains

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

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
    ("system", "You are a standup comedian who tells hilarious jokes on {topic}"),
    ("human", "Tell me {joke_count} jokes"),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

# create our runnable lambdas
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# now build a chain
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words
# and invoke it
response = chain.invoke({"topic": "Rahul Gandhi", "joke_count": "1"})
console.print(f"[green]Chain output:[/green]\n{response}")
