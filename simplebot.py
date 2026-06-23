from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # use to store secret

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]
llm = ChatOpenAI(model="gpt-4o")

def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state
graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()
user_input= input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")

#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



