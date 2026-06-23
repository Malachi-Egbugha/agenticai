from typing import Dict, TypedDict, List
import random
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    name: str
    number: List[int]
    counter: int

def greeting_node(state: AgentState) -> AgentState:
    """greeting node which says hi to the person"""
    state["name"] =f"Hi there, {state["name"]}"
    state["counter"] = 0
    return state
def random_node(state: AgentState) -> AgentState:
    """generate a random node from 0 to 10"""
    state["number"].append(random.randint(1, 10))
    state["counter"] += 1
    return state
def should_continue(state: AgentState) -> AgentState:
    """function should decide what to do next"""
    if state["counter"] <= 5:
        print("entering loop", state["counter"])
        return "loop" #continue looping
    else:
        return "exit" #exit the loop

graph = StateGraph(AgentState)
graph.add_node("greeting", greeting_node)
graph.add_node("random", random_node)
graph.add_edge("greeting", "random")
graph.add_conditional_edges(
    "random",
    should_continue,
    {
        "loop": "random", #Self loop back to same node
        "exit": END #End graph
    }
)

graph.set_entry_point("greeting")
app = graph.compile()
# Print the graph directly in the PyCharm terminal
#print(app.get_graph().draw_ascii())
#initial_state_1 = AgentState(name="Malachi", number=[], counter=-1)
print(app.invoke({"name":"Malachi", "number":[], "counter":-1}))
