"""
prompt_template_basic.py - basic prompt template with LangChain & Gemi

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_core.prompts import ChatPromptTemplate

# load all environment variables
load_dotenv()

# only for colorful text & markdown output support
console = Console()

# create out prompt template
prompt_str = "Tell me a joke about [{topic}]"
prompt_template = ChatPromptTemplate.from_template(prompt_str)
# console.print(f"Prompt template: [yellow]{prompt_template}[/yellow]")
prompt = prompt_template.invoke({"topic": "Rahul Gandhi"})
console.print(f"Prompt (with 1 param): [yellow]{prompt}[/yellow]")

# you can have multiple variables in a PromptTemplate
prompt_str = "Tell me a {type_of_joke} joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(prompt_str)
prompt = prompt_template.invoke({"type_of_joke": "political", "topic": "Rahul Gandhi"})
console.print(f"Prompt (with multiple params): [green]{prompt}[/green]")

# if you want to use Prompt templates & define message types
messages = [
    ("system", "You are a stand-up comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_count} jokes"),
]
# NOTE: instead of "from_template", you now call "from_messages"
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "Rahul Gandhi", "joke_count": "3"})
console.print(f"Prompt (with templates & message types): [blue]{prompt}[/blue]")
