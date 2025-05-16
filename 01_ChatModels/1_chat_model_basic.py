"""
chat_model_basic.py - basic chat model structure with LangChain
    & Google Gemini

@Author: Manish Bhobe
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_google_genai import ChatGoogleGenerativeAI

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
console = Console()

response = model.invoke("What is 81 divided by 9")
console.print(f"Full response:")
console.print(response)
console.print(f"Just the content: {response.content}")
