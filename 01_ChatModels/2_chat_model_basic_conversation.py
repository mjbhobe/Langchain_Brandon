from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# load all environment variables
load_dotenv()

# create my LLM - using Google Gemini
# model = ChatOpenAI(
#     model="gpt-4o",
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
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
    HumanMessage("What is area of circle with radius 4.5?"),
]

# and invoke the model with these messages
response = model.invoke(messages)
console.print(Markdown(response.content))
