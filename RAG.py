from dotenv import load_dotenv # use to store secret
import os
import requests
import json
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,ToolMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#Our Embedding Model has to be also be compatible with LLM

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

pdf_path = "Stock.pdf"

# Safety measure I have out for debugging purposes
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) #This loads the pdf

#Check if the pdf is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise
#Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages) #We now apply this to our pages
persist_directory =r"C:\Users\user\Downloads\crm\db"
collection_name = "stock_market"

#if our collection does not exist in the directory , we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
try:
    # Here, we actually create the chroma database using our ambeddings model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name

    )
    print("Created Chroma vector store")
except Exception as e:
    print(f"Error setting up ChromaDb: {str(e)}")
    raise

#Now we create our retriever
retriever = vectorstore.as_retriever(
    search_type ="similarity",
    search_kwargs ={"k": 5} #k is the amount of chunks to return
)

@tool()
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance document
    :param query:
    :return:
    """
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the Stock performance 2024 document"
    result =[]
    for i, doc in enumerate(docs):
        result.append(f"Document {i + 1}: \n {doc.page_content}")
    return "\n\n".join(result)

@tool
def get_customer_bills(customer_id: str) -> str:
    """
    Fetch customer billing history, payment history, outstanding balance, and customer details.

    Args:
        customer_id: The unique customer account number (e.g., "719721740").
    """
    api_key = os.getenv("AGENT_API_KEY", "Le7JEEWjFA8MCfzQWair3WlT")
    url = f"http://196.6.217.90:3009/api/agent/bills?id={customer_id}"
    try:
        response = requests.get(url, headers={"apikey": api_key}, timeout=10)
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Failed to retrieve bills. Status code: {response.status_code}. Error: {response.text}"
    except Exception as e:
        return f"Error connecting to bills API: {str(e)}"

@tool
def get_customer_statement(customer_id: str) -> str:
    """
    Fetch the full account statement for a customer, including transaction history and account summary.

    Args:
        customer_id: The unique customer account number (e.g., "719721740").
    """
    api_key = os.getenv("AGENT_API_KEY", "Le7JEEWjFA8MCfzQWair3WlT")
    url = f"http://196.6.217.90:3009/api/agent/statement?id={customer_id}"
    try:
        response = requests.get(url, headers={"apikey": api_key, "accept": "application/json"}, timeout=10)
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Failed to retrieve statement. Status code: {response.status_code}. Error: {response.text}"
    except Exception as e:
        return f"Error connecting to statement API: {str(e)}"

@tool
def get_employee(employee_id: str) -> str:
    """
    Fetch details for an employee by their employee ID.

    Args:
        employee_id: The unique employee ID number (e.g., "150204").
    """
    api_key = os.getenv("AGENT_API_KEY", "Le7JEEWjFA8MCfzQWair3WlT")
    url = f"http://196.6.217.90:3009/api/agent/employee/{employee_id}"
    try:
        response = requests.get(url, headers={"apikey": api_key, "accept": "application/json"}, timeout=10)
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Failed to retrieve employee. Status code: {response.status_code}. Error: {response.text}"
    except Exception as e:
        return f"Error connecting to employee API: {str(e)}"

@tool
def get_customer_tokens(customer_id: str) -> str:
    """
    Fetch the token/vending history for a prepaid customer, including token codes and amounts purchased.

    Args:
        customer_id: The unique customer account number (e.g., "719721740").
    """
    api_key = os.getenv("AGENT_API_KEY", "Le7JEEWjFA8MCfzQWair3WlT")
    url = f"http://196.6.217.90:3009/api/agent/tokens?id={customer_id}"
    try:
        response = requests.get(url, headers={"apikey": api_key, "accept": "application/json"}, timeout=10)
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Failed to retrieve tokens. Status code: {response.status_code}. Error: {response.text}"
    except Exception as e:
        return f"Error connecting to tokens API: {str(e)}"

tools = [retriever_tool, get_customer_bills, get_customer_statement, get_employee, get_customer_tokens]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages ]
def should_continue(state: AgentState):
    """Check if the last message contains tool call"""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0
system_prompt="""
You are an intelligent 
"""
tools_dict = {our_tool.name: our_tool for our_tool in tools} #create a dictionary of our tools

#LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}

#retriever agent
def take_action(state: AgentState):
    """Execute tool calls from the LLMs response"""
    tool_calls = state["messages"][-1].tool_calls
    results =[]
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with args: {t['args']}")
        if not t["name"] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist")
            result = "Incorrect Tool Name, Please Retry and select tool from list of Available tools"
        else:
            result = tools_dict[t["name"]].invoke(t['args'])
            print(f"Result length: {len(str(result))}")

        #Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'],name=t['name'], content=str(result)))
    print("Tools execution Complete back to the model")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()









