"""
chain_basics.py - basics of LangChain chains.

@Author: Manish Bhobe
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

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

# now chain the prompt template + model + output parser
chain = prompt_template | model | StrOutputParser()
# and invoke it with appropriate values for prompt template variables
params = {"topic": "Rahul Gandhi", "joke_count": "3"}
response = chain.invoke(params)
console.print(
    f"[red] --- Showing {params['joke_count']} LLM generated jokes on {params['topic']} ---[/red]"
)
# NOTE: since I have StrOutputParser() at the end of the chain, my response
# is a string - so don't need response.content
console.print(f"[green] LLM Response [/green]: \n {response}")
