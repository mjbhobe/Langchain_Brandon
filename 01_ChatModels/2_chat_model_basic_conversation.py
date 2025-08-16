"""
chat_model_basic_conversation.py - extending the basic Q&A chat model to
    simiulate a conversation with SystemMessage, HumanMessage & AIMessage
    using LangChain & Google Gemini.

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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

# setup the messages
messages = [
    SystemMessage("Solve the following problem."),
    HumanMessage("What is 81 divided by 9?"),
]

# and invoke the model with these messages
response = model.invoke(messages)
console.print(response.content)

# simulate a conversation (as you would in ChatGPT)
messages = [
    SystemMessage("Solve the following problem."),
    # question I ask
    HumanMessage("What is 81 divided by 9?"),
    # response I get from LLM
    AIMessage("81 divided by 9 is 9."),
    # next question I ask (expecting ~63.62)
    # HumanMessage("What is area of circle with radius 4.5?"),
    # since LLM has complete context, it should respond with 9 x 3 = 27 (or something similar)
    HumanMessage("What is answer of previous question multiplied by 3?"),
]

# and invoke the model with these messages
response = model.invoke(messages)
console.print(Markdown(response.content))
