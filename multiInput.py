from typing import Dict, TypedDict,List
from langgraph.graph import StateGraph, state

class AgenticState(TypedDict):
    values: List[int]
    names: str
    result: str

def process_values(state: AgenticState) -> AgenticState:
    """ this function handles multiple different inputs"""