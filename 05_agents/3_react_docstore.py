"""
3_react_docstore.py - chatting with a PDF document

@author: Manish Bhobe
My experiments with Python, AI/ML and Gen AI.
Code is shared for learning purposed only - use at own risk!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
console = Console()


# function to tell time
def get_current_time(*args, **kwargs) -> str:
    """Returns the current time in H:MM AM/PM format."""
    from datetime import datetime

    return datetime.now().strftime("%I:%M %p")


def search_wikipedia(query: str) -> str:
    """Searches Wikipedia and returns summary of the first result."""
    # import wikipedia

    # try:
    #     wikipedia.session.headers.update(
    #         {"User-Agent": "MyCoolBot/1.0 (https://mywebsite.com/)"}
    #     )
    #     return wikipedia.summary(query, sentences=2)
    import requests

    try:
        headers = {"User-Agent": "MyCoolBot/1.0 (https://mywebsite.com/)"}
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"

        response = requests.get(url, headers=headers)
        data = response.json()
        return data["extract"]
    except Exception as e:
        return f"Error searching Wikipedia for: {str(e)}"


web_search = TavilySearchResults(k=5)


tools = [
    # get current time tool
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="Get the current time in HH:MM AM/PM format.",
    ),
    # # search Wikipedia tool
    # Tool(
    #     name="Wikipedia",
    #     func=search_wikipedia,
    #     description="Useful when you want to know information about a topic.",
    # ),
    # search web other than wikipedia
    Tool(
        name="Tavily Search",
        func=web_search.run,
        description="""Useful for answering questions about current events or recent 
            facts by searching the web""",
    ),
]

# create your model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# your prompt
prompt = hub.pull("hwchase17/structured-chat-agent")
console.print(f"Prompt: {prompt}")

# create conversational memory for the chat agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# create the agent
agent = create_structured_chat_agent(
    llm=model,
    prompt=prompt,
    tools=tools,
    stop_sequence=True,
)

# execute the agent
agent_executer = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# setup system message
initial_message: str = (
    """You are a helpful assistant. You can answer questions and perform tasks like telling the current time or searching Wikipedia or searching the web. If you are unable to answer the question, respond with a \"I am unable to answer that question\" message"""
)
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# indefinate conversation until user types "exit" or "bye" or "quit"
console = Console()
while True:
    console.print("\n[bold green]You: [/bold green]", end="")
    user_query = input()
    if user_query.strip().lower() in ["exit", "bye", "quit"]:
        console.print("[bold red]Exiting the chat...[/bold red]")
        break

    memory.chat_memory.add_message(HumanMessage(content=initial_message))
    # now ask our agent executer to run our query
    response = agent_executer.invoke({"input": user_query})
    console.print(f"[yellow]Bot: [/yellow]\n{response['output']}")
    # and save response in chat memory too
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
