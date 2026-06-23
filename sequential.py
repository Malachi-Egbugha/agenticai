from typing import Dict, TypedDict
from langgraph.graph import StateGraph, state


class AgentState(TypedDict):
    name: str
    age: str
    final:str
def firstnode(state: AgentState) -> AgentState:
    """this is the first node of our sequence"""
    state["final"]=f"Hi {state["name"]}"
    return state
def secondnode(state: AgentState) -> AgentState:
    """this is the second node of our sequence"""
    state["final"]= state["final"] + f"You are {state["age"]} years old"
    return state
graph = StateGraph(AgentState)
graph.add_node("firstnode", firstnode)
graph.add_node("secondnode", secondnode)
graph.set_entry_point("firstnode")
graph.add_edge("firstnode", "secondnode")
graph.set_finish_point("secondnode")
app= graph.compile()



