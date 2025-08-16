"""
chat_model_alternatives.py - illustrates use of other LLM providers, such as
    OpenAI ChatGPT or Anthropic Claude. You'll need API keys for both these
    providers.

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

import random
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# load all environment variables
load_dotenv()

SUPPORTED_PROVIDERS = ["openai", "anthropic", "google"]


def get_model(provider: str):

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported {provider}! Choose one of {SUPPORTED_PROVIDERS}")

    if provider == "openai":
        model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )
    elif provider == "anthropic":
        model = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            max_tokens=2048,
            timeout=None,
            max_retries=2,
            # other params...
        )
    elif provider == "google":
        # create my LLM - using Google Gemini
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )
    else:
        raise ValueError(f"{provider} is unsupported! Pick one of openai|claude|gemini")
    return model


# randomly pick a provider
provider_index = random.choice(range(len(SUPPORTED_PROVIDERS)))
print(f"Randomly chosen provider: {SUPPORTED_PROVIDERS[provider_index]}")
model = get_model(SUPPORTED_PROVIDERS[provider_index])

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
