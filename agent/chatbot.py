"""
agent/chatbot.py

The operator-facing chatbot. Built on LangChain's current (v1.0+)
create_agent() API, which replaced the deprecated AgentExecutor /
create_tool_calling_agent pattern. create_agent() runs on LangGraph
under the hood and returns a compiled graph you call .invoke() on.

Three tools:

  1. predict_tool   -> runs the existing TFT model + explainability
                        (reuses model/tft.py and evaluation/explain.py,
                        no new ML logic)
  2. graph_tool      -> queries Neo4j for similar past faults and how
                        they were resolved (knowledge/graph_writer.py)
  3. retrieve_tool   -> semantic search over maintenance procedure docs
                        (knowledge/rag_store.py)

The LLM's job is orchestration + synthesis of what these three tools
return into one operator-readable answer -- it is not asked to reason
about sensor physics on its own, which keeps answers grounded in your
actual model output and documents instead of hallucinated.

Requires: pip install langchain langchain-openai
  (or swap the model string for a local Ollama model -- see build_agent())
Env vars: OPENAI_API_KEY (only if using an OpenAI model string)
"""
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.agents import create_agent
from langchain_core.tools import tool

from model.tft import TemporalFusionTransformer
from evaluation.explain import extract_sensor_importance
from knowledge.graph_writer import get_graph_store
from knowledge.rag_store import retrieve_procedures

# ------------------------------------------------------------
# Model loading (reuses the same checkpoint api.py serves)
# ------------------------------------------------------------
_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = TemporalFusionTransformer(
            input_dim=68, hidden_dim=128, num_heads=8, dropout=0.068, num_classes=2
        )
        _MODEL.load_state_dict(
            torch.load("data/processed/best_model.pt", map_location="cpu")
        )
        _MODEL.eval()
    return _MODEL


# ------------------------------------------------------------
# TOOL 1 -- prediction + explainability
# ------------------------------------------------------------
@tool
def predict_tool(machine_id: str, window_index: int = -1) -> dict:
    """Run the fault-prediction model on a sensor window for the given
    machine_id and return the fault/normal label, confidence, and the
    top-3 sensors that drove the prediction. Use this whenever the
    operator asks about a machine's current status or why it was flagged.

    window_index selects which window from the held-out test set to
    evaluate: -1 (default) uses the most recent window, or pass a
    specific non-negative index to inspect a particular historical
    window (useful for demos/debugging). In a live deployment this
    would instead pull the latest 60-step window from a real-time
    sensor feed rather than indexing into a static test set."""
    model = _load_model()

    X_test = np.load("data/processed/X_test.npy")
    idx = window_index if window_index >= 0 else len(X_test) + window_index
    idx = max(0, min(idx, len(X_test) - 1))
    x = torch.tensor(X_test[idx:idx + 1], dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(logits.argmax(dim=1).item())

    sensor_scores, _, top3 = extract_sensor_importance(model, x)

    result = {
        "machine_id": machine_id,
        "window_index": idx,
        "label": "NORMAL" if pred == 1 else "FAULT",
        "confidence": round(float(probs[pred]), 4),
        "top_sensors": [
            {"rank": i + 1, "sensor": s, "importance": round(v, 4)}
            for i, (s, v) in enumerate(top3)
        ],
    }

    if result["label"] == "FAULT":
        gs = get_graph_store()
        event_id = gs.log_fault_event(
            machine_id=machine_id,
            confidence=result["confidence"],
            top_sensors=result["top_sensors"],
            model_version="tft-v1",
        )
        result["event_id"] = event_id

    return result


# ------------------------------------------------------------
# TOOL 2 -- knowledge graph query
# ------------------------------------------------------------
@tool
def graph_tool(sensor_ids: list) -> list:
    """Given a list of sensor IDs (e.g. ['PS1', 'TS2']) implicated in a
    current fault, query the knowledge graph for past fault events that
    were flagged by an overlapping set of sensors, and how each one was
    resolved. Use this to answer 'has this happened before' or 'what
    fixed it last time'."""
    gs = get_graph_store()
    return gs.find_similar_past_faults(sensor_ids, min_overlap=1, limit=5)


# ------------------------------------------------------------
# TOOL 3 -- RAG over maintenance procedures
# ------------------------------------------------------------
@tool
def retrieve_tool(query: str) -> list:
    """Semantic search over the maintenance procedure documents. Use this
    to find the recommended diagnostic steps for a given sensor or failure
    mode, e.g. 'what to check for a PS1 pressure anomaly'."""
    return retrieve_procedures(query, k=3)


TOOLS = [predict_tool, graph_tool, retrieve_tool]

SYSTEM_PROMPT = """You are a logistics/maintenance operations assistant for a \
predictive maintenance system monitoring hydraulic equipment.

When an operator asks about a machine:
1. Use predict_tool to get the current model prediction and the sensors \
that drove it.
2. If a fault is detected, use graph_tool with those sensor IDs to check \
whether similar faults happened before and how they were resolved.
3. Use retrieve_tool to pull the relevant diagnostic procedure for the \
flagged sensor(s).
4. Synthesize all three into one clear, actionable answer for a \
maintenance operator: what was detected, why (which sensors), whether \
this has happened before and what fixed it, and what to check first per \
the procedure docs.

Be concise and concrete. Do not invent sensor readings, past events, or \
procedures that the tools did not return -- if a tool returns nothing \
relevant, say so plainly rather than guessing."""


def build_agent(model=None):
    """Build the compiled agent graph.

    `model` can be:
      - omitted, to default to an OpenAI model string (requires
        OPENAI_API_KEY and `pip install langchain-openai`)
      - a provider:model string, e.g. "anthropic:claude-sonnet-4-6" or
        "ollama:llama3" for a fully local/free setup (requires
        `pip install langchain-ollama` and `ollama pull llama3`)
      - an already-constructed BaseChatModel instance

    Returns a compiled LangGraph agent -- call .invoke({"messages": [...]})
    on it, not .run() or .invoke({"input": ...}) as in the old API.
    """
    if model is None:
        model = "openai:gpt-4o-mini"

    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def ask(agent, message: str) -> str:
    """Convenience wrapper: send one user message, get the final text reply.
    Handles the {"messages": [...]} in/out shape so callers (e.g. the
    FastAPI /chat endpoint) don't need to know the internal message format."""
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = build_agent()
    reply = ask(agent, "Why did machine HYD-01 get flagged, and what should I check first?")
    print(reply)
