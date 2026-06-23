from multiprocessing import context
from typing import Annotated, TypedDict, Sequence

import langchain_core
from langchain_core import tools
from langchain_core.messages import BaseMessage, \
    HumanMessage  # The foundational class for all message types in langGraph
from langchain_core.messages import ToolMessage # Passes data back to LL< after it calls a tool such as the content
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv # use to store secret

from simplebot import user_input

load_dotenv()

#this is the global variable to store document content
document_content =""

class AgentState(TypedDict):
    messages: Annotated[Sequence(BaseMessage), add_messages]

@tool
def update(content:str) -> str:
    """update the document with the  content"""
    global document_content
    document_content = content
    return f"Document has been updated sucessfully! The current content is: {document_content}"

@tool
def save(filename:str) -> str:
    """save the document to a text file and  finish the process

    Args:
        filename: Name for the tesxt file
    """
    global document_content
    if not filename.endswith("txt"):
        filename = f"{filename}.txt"
    try:
        with open(filename,"w") as file:
            file.write(document_content)
        print(f"\n Document has been saved sucessfully to : {filename}")
        return f"Document has been saved sucessfully to : {filename}"
    except Exception as e:
        return f"Error saving document: {str(e)}"

tools=[update, save]
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)
def our_agent(state:AgentState) -> AgentState:
    system_prompt=SystemMessage(content=f"""
    You are drafter, a helpful  
    """)
    if not state["messages"]:
        user_input="I am ready to help you update a document. what will you like to create?"
        user_message= HumanMessage(content=user_input)
    else:
        user_input=input("\n What will you like to do with the document?")
        print(f"\n USER: {user_input}")
        user_message= HumanMessage(content=user_input)
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)