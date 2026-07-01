"""
api.py — FastAPI bridge to the LangGraph RAG agent (RAG.py).

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("rag_api")


# ---------------------------------------------------------------------------
# Lifespan: load the compiled LangGraph agent once at startup
# ---------------------------------------------------------------------------
_agent_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG agent graph at startup; clean up on shutdown."""
    logger.info("Loading RAG agent …")
    try:
        # Deferred import so FastAPI itself starts even if RAG.py has issues
        from RAG import rag_agent  # noqa: PLC0415
        _agent_state["graph"] = rag_agent
        logger.info("RAG agent loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load RAG agent: %s", exc)
        # Store exception so the dependency can surface a clear 503
        _agent_state["error"] = str(exc)
    yield
    _agent_state.clear()
    logger.info("RAG agent unloaded.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Agent API",
    description=(
        "A production-ready FastAPI gateway to the LangGraph-powered RAG agent. "
        "Send a natural-language query and receive the agent's final response."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — tighten `allow_origins` in production to your actual front-end origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred.", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """Incoming user query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The natural-language question or instruction for the agent.",
        examples=["What is the outstanding balance for customer 719721740?"],
    )


class AgentResponse(BaseModel):
    """Final response returned by the agent."""

    answer: str = Field(..., description="The agent's final text response.")
    tool_calls_made: list[str] = Field(
        default_factory=list,
        description="Names of tools the agent called while answering.",
    )


# ---------------------------------------------------------------------------
# Dependency injection — provides the compiled graph
# ---------------------------------------------------------------------------
def get_graph():
    """Dependency that yields the compiled LangGraph agent.

    Raises:
        HTTPException 503: if the agent failed to load at startup.
    """
    if "error" in _agent_state:
        raise HTTPException(
            status_code=503,
            detail=f"Agent unavailable: {_agent_state['error']}",
        )
    if "graph" not in _agent_state:
        raise HTTPException(status_code=503, detail="Agent not yet initialised.")
    return _agent_state["graph"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Liveness probe — returns 200 when the service is up."""
    ready = "graph" in _agent_state
    return {
        "status": "ok" if ready else "degraded",
        "agent_loaded": ready,
    }


@app.post(
    "/query",
    response_model=AgentResponse,
    summary="Query the RAG agent",
    tags=["Agent"],
)
async def query_agent(
    request: QueryRequest,
    graph=Depends(get_graph),
) -> AgentResponse:
    """
    Invoke the LangGraph RAG agent with the user's query.

    The agent will autonomously decide which tools to call (document retrieval,
    customer bills, statements, tokens, employee lookup, etc.) before returning
    its final answer.
    """
    logger.info("Received query: %.120s", request.query)

    initial_state = {
        "messages": [HumanMessage(content=request.query)]
    }

    try:
        # Run the synchronous LangGraph graph in a thread-pool so the event
        # loop stays non-blocking (ainvoke is not yet universally supported
        # for custom sync nodes).
        final_state = await asyncio.get_event_loop().run_in_executor(
            None, graph.invoke, initial_state
        )
    except Exception as exc:
        logger.exception("Agent execution failed")
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution error: {str(exc)}",
        ) from exc

    messages = final_state.get("messages", [])

    # Extract the last AI message as the answer
    answer = ""
    for msg in reversed(messages):
        # AIMessage has .content; skip ToolMessages / HumanMessages
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            answer = msg.content
            break

    if not answer:
        raise HTTPException(
            status_code=500,
            detail="Agent returned no usable response.",
        )

    # Collect names of every tool that was called during the run
    tool_calls_made: list[str] = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc.get("name", "unknown"))

    logger.info(
        "Query answered. Tools used: %s | Answer length: %d chars",
        tool_calls_made or "none",
        len(answer),
    )

    return AgentResponse(answer=answer, tool_calls_made=tool_calls_made)
