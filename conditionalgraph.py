from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    number1: int
    operator: str
    number2: int
    finalNumber: int
def adder(state: AgentState) -> AgentState:
    """this node adds the two numbers"""
    state["finalNumber"]= state["number1"] + state["number2"]
    return state
def subtractor(state: AgentState) -> AgentState:
    """this node subtract the two numbers  """
    state["finalNumber"]= state["number1"] - state["number2"]
    return state
def decide_next_node(state: AgentState) -> AgentState:
    """this node decides if the next node of the graph"""
    if (state["operator"] == "+"):
        return "addition_operation"
    elif (state["operator"] == "-"):
        return "subtraction_operation"

graph = StateGraph(AgentState)
graph.add_node("add_node", adder)
graph.add_node("subtract_node", subtractor)
graph.add_node("router", lambda state: state)
graph.add_edge(START, "router")
graph.add_conditional_edges("router", decide_next_node, {
    # Edge: Node
    "additional_operation": "add_node",
    "subtraction_operation": "subtract_node"
})

graph.add_edge("add_node", END)
graph.add_edge("subtract_node", END)
app = graph.compile()
# Print the graph directly in the PyCharm terminal
print(app.get_graph().draw_ascii())
initial_state_1 = AgentState(number1=10, operator="-", number2=5)
print(app.invoke(initial_state_1))  