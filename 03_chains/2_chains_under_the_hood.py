"""
chain_under_the_hood.py - a look at RunnableLambda & RunnableSequence
    to understand how chains work under the hood

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
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
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)
upper_case_output = RunnableLambda(lambda x: x.upper())

# and this is what LangChain does under the hood
# first & last are single items, middle can be a list of items (many items)
chain = RunnableSequence(
    first=format_prompt, middle=[invoke_model, parse_output], last=upper_case_output
)
response = chain.invoke({"topic": "Rahul Gandhi", "joke_count": "3"})
console.print(f"[green] AI Response [/green]\n {response}")
