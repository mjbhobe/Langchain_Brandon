"""
chat_model_save_messages_history_firebase.py - console based ChatGPT like application
    using chat history, SystemMessage, HumanMessage & AIMessage in Langchain
    & Google Gemini LLM. Chat history is saved to Firebase.

@Author: Manish Bhobe
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

"""
Steps to replicate this example:
1. Create a Firebase Account
2. Create a Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate Google Client with your Google Account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to new Firebase project you just created
5. Enable Firebase API in the Google Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=<project_name>
"""

# load all environment variables
load_dotenv()

PROJECT_ID = "chatwithgemini-b850c0"
SESSION_ID = "user_session"
COLLECTION_NAME = "chat_history"

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

# this is our chat history
chat_history = []

# setup the messages
message = SystemMessage(
    content="""You are a helpful AI assistant that can anwser questions
    on a variety of subjects"""
)
chat_history.append(message)

# endless loop unless user types "quit" or "exit" or "bye"
while True:
    console.print("[blue]Your query?[/blue]")
    query = input().strip().lower()
    if query in ["quit", "exit", "bye"]:
        break
    human_message = HumanMessage(content=query)
    chat_history.append(human_message)
    console.print("[yellow]AI generating...[/yellow]", end="")

    # pass the entire history to LLM
    response = model.invoke(chat_history)
    chat_history.append(response)
    console.print("\r[green]AI response:[/green]")
    console.print(Markdown(response.content))

# finish up by showing the entire chat history
console.print("[red] ----------- Chat history ----------- [/red]")
console.print(chat_history)
