"""
prompt_template_with_chatmodel.py - here we will interface the prompt with a chat model
    (Google Gemini) and get a response.

@Author: Manish Bhobe
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

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

# create out prompt template
console.print("[red] ------ first test ------ [/red]")
prompt_str = "Tell me a joke about [{topic}]"
prompt_template = ChatPromptTemplate.from_template(prompt_str)
# console.print(f"Prompt template: [yellow]{prompt_template}[/yellow]")
prompt = prompt_template.invoke({"topic": "Rahul Gandhi"})
console.print(f"Prompt (with 1 param): [yellow]{prompt}[/yellow]")
response = model.invoke(prompt)
console.print(f"[green]Model Response:[/green]\n{response.content}")

# you can have multiple variables in a PromptTemplate
console.print("[red] ------ second test ------ [/red]")
prompt_str = "Tell me a {type_of_joke} joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(prompt_str)
prompt = prompt_template.invoke(
    {"type_of_joke": "embarassing", "topic": "Rahul Gandhi"}
)
console.print(f"Prompt (with multiple params): [green]{prompt}[/green]")
response = model.invoke(prompt)
console.print(f"[green]Model Response:[/green]\n{response.content}")


# if you want to use Prompt templates & define message types
console.print("[red] ------ third test ------ [/red]")
messages = [
    ("system", "You are a stand-up comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_count} {adjective} jokes"),
]
# NOTE: instead of "from_template", you now call "from_messages"
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke(
    {"topic": "Rahul Gandhi", "joke_count": "3", "adjective": "ROFLMAO"}
)
console.print(f"Prompt (with templates & message types): [blue]{prompt}[/blue]")
response = model.invoke(prompt)
console.print(f"[green]Model Response:[/green]\n{response.content}")
