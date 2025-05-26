"""
chains_parallel.py - run chains in parallel

@Author: Manish Bhobe
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

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

# messages for PromptTemplate (NOTE: SystemMessage must be first in the list
messages = [
    (
        "system",
        """
        You are an expert product reviewer who can explain features in a simple language, 
        clearly articulating the pros and cons of a product.
        """,
    ),
    ("human", "List all the main features and standouts of {product}"),
]
prompt_template = ChatPromptTemplate.from_messages(messages)


def analyze_pros_prompt(product_features: str):
    pros_messages = [
        ("system", "You are an expert product reviewer"),
        (
            "human",
            "Given these product features: {features}, list all the pros of the features. Do not add any additional text or explanations.",
        ),
    ]
    pros_template = ChatPromptTemplate.from_messages(pros_messages)
    return pros_template.format_prompt(features=product_features)


def analyze_cons_prompt(product_features: str):
    cons_messages = [
        ("system", "You are an expert product reviewer"),
        (
            "human",
            "Given these product features: {features}, list all the cons of the features. Do not add any additional text or explanations.",
        ),
    ]
    cons_template = ChatPromptTemplate.from_messages(cons_messages)
    return cons_template.format_prompt(features=product_features)


# create our runnable lambdas
pros_branch_chain = (
    RunnableLambda(lambda product_features: analyze_pros_prompt(product_features))
    | model
    | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda product_features: analyze_cons_prompt(product_features))
    | model
    | StrOutputParser()
)


def combine_pros_and_cons(pros, cons):
    return f"**Pros:**\n{pros}\n\n**Cons:**\n{cons}"


# now build a chain
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(
        lambda x: combine_pros_and_cons(x["branches"]["pros"], x["branches"]["cons"])
    )
)
# and invoke it
product = "iPhone 15"
response = chain.invoke({"product": product})

from rich.markdown import Markdown

console.print(f"[blue]Here are the pros & cons of **{product}**[/blue]\n")
console.print(Markdown(response))
