"""
chains_branching.py - example of chain branches depending on initial outcome

@Author: Manish Bhobe
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch

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

# template to use if feedback is positive!
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a thank-you note for this positive feedback: {feedback}"),
    ]
)

# template to use if feedback is negative!
negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a response addressing this negative feedback: {feedback}"),
    ]
)

# template to use if feedback is neutral!
neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}",
        ),
    ]
)

# template to use if feedback requures and escalation
escalation_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "human",
            "Generate a message to escalate this feedback to a human agant: {feedback}",
        ),
    ]
)

# template to use to classify the sentiment of the feedback/review
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "human",
            "Classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}",
        ),
    ]
)

# the chain that classifies sentiment of the feedback
classification_branch = classification_template | model | StrOutputParser()

# branching off to various handlers depending on sentiment of feedback
routing_branches = RunnableBranch(
    (
        lambda x: "positive" in x.lower(),
        positive_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: "negative" in x.lower(),
        negative_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: "neutral" in x.lower(),
        neutral_feedback_template | model | StrOutputParser(),
    ),
    # for all else, escalate
    escalation_feedback_template | model | StrOutputParser(),
)

# chain that ties them together
chain = classification_branch | routing_branches

# here are sample feedbacks
positive_feedback = """
I had a great experience with CleenoBot. It did a fantastic job of cleaning all floors, especially around the corners and in hard to reach places. I would highly recommend it to others. Thank you for such a wonderful product!"""

negative_feedback = """
I has high expectations of CleenoBot, especially after seeing the demo. However I am dissappointed with the quality of cleaning. It hardly picks up any dust or dirt off the carpet and can hardly reach the corners of my room. I am returning this & will not recommend this to anyone."""

neutral_feedback = """
The CleenoBot is okay, but it doesn't do anything special. It cleans the floors, but I expected more features for the price. It's not bad, but it's not great either."""

escalation_feedback = """
I am not happy with the CleenoBot. It does not clean my floors properly and I have tried to contact support but have not received a response. I'd like to speak to your manager - want my money back. This is cheating!!!"""

# console.print(f"[green]Positive Review:[/green] {positive_feedback}\n")
# response = chain.invoke({"feedback": positive_feedback})
# console.print(Markdown(response))

# console.print(f"[yellow]Negative Review:[/yellow] {negative_feedback}\n")
# response = chain.invoke({"feedback": negative_feedback})
# console.print(Markdown(response))

# console.print(f"[blue]Neutral Review:[/blue] {neutral_feedback}\n")
# response = chain.invoke({"feedback": neutral_feedback})
# console.print(Markdown(response))

console.print(f"[red]Escalation Review:[/red] {escalation_feedback}\n")
response = chain.invoke({"feedback": escalation_feedback})
console.print(Markdown(response))
