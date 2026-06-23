from typing import Dict, TypedDict
from langgraph.graph import StateGraph, state


class AgentState(TypedDict):
    message: str

