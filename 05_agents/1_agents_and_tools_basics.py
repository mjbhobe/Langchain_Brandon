"""
1_agents_and_tools_basics.py - basic agent with time tool

@author: Manish Bhobé
My experiments with Python, AI/ML and Gen AI.
Code is shared for learning purposed only - use at own risk!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
console = Console()


# function to tell time
def get_current_time(*args, **kwargs) -> str:
    """Returns the current time in H:MM AM/PM format."""
    from datetime import datetime

    return datetime.now().strftime("%I:%M %p")


tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="Get the current time in HH:MM AM/PM format.",
    )
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
prompt = hub.pull("hwchase17/react")
console.print(f"Prompt: {prompt}")

# create the agent
agent = create_react_agent(
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
)

# execute the agents
response = agent_executer.invoke({"input": "What is the current time?"})
console.print(f"[green]Response: [/green] {response['output']}")
