# Integration Guide — Knowledge Graph + RAG + Chatbot

This wires the new `knowledge/` and `agent/` modules into the existing
FastAPI app (`api.py`) with one new endpoint. Nothing in the existing
model, training, or MLOps code changes.

## 1. Install dependencies

```bash
pip install -r requirements.txt -r requirements_agent.txt
```

## 2. Start Neo4j

Easiest path is the official Docker image (no local install needed):

```bash
docker run -d --name neo4j-maintenance \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password_here \
  neo4j:5
```

Then load the schema once:

```bash
cat knowledge/schema.cypher | docker exec -i neo4j-maintenance \
  cypher-shell -u neo4j -p your_password_here
```

Or paste `knowledge/schema.cypher` into the Neo4j Browser at
`http://localhost:7474`.

## 3. Set environment variables

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password_here"
export OPENAI_API_KEY="sk-..."   # only if using ChatOpenAI — see note below
```

**Fully free / local alternative:** swap `ChatOpenAI` in
`agent/chatbot.py`'s `build_agent()` for
`langchain_community.chat_models.ChatOllama(model="llama3")` after
`ollama pull llama3` — no API key or cost, runs on your own machine.

## 4. Build the RAG index once

```bash
python -m knowledge.rag_store
```

This embeds the docs in `procedures/` and saves a FAISS index to
`knowledge/faiss_index/` so it doesn't need to be rebuilt on every
process start.

## 5. Add the `/chat` endpoint to `api.py`

Add this near your other endpoint definitions (after the existing
`/predict` endpoint is a natural place):

```python
# ── Add these imports near the top of api.py ─────────────────────
from agent.chatbot import build_agent

# ── Add this to the lifespan() function, alongside load_model() ──
_AGENT_EXECUTOR = None

@asynccontextmanager
async def lifespan(app):
    global _AGENT_EXECUTOR
    load_model()
    load_onnx_session()
    _AGENT_EXECUTOR = build_agent()   # <-- add this line
    yield

# ── New request/response schemas ──────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# ── New endpoint ────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
def chat(data: ChatRequest):
    """Operator-facing chatbot: prediction + knowledge graph + RAG,
    synthesized by an LLM via LangChain."""
    result = _AGENT_EXECUTOR.invoke({"input": data.message})
    return ChatResponse(reply=result["output"])
```

That's the whole integration — the agent reuses `model/tft.py` and
`evaluation/explain.py` exactly as they already exist; nothing about
the model itself changes.

## 6. Try it

```bash
python api.py
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Why did machine HYD-01 get flagged, and what should I check first?"}'
```

## 7. Optional: add a chat panel to the existing `/ui` dashboard

`templates/index.html` already has a stats bar and prediction panel —
a simple text input + message list calling `POST /chat` fits the
existing dark-industrial theme without a redesign. This is a good
"nice to have" if there's time, but the API endpoint above is the part
that actually demonstrates the architecture in an interview.

## What to say about this in an interview

- The graph is populated automatically as a side effect of real
  predictions (`predict_tool` calls `log_fault_event` whenever the
  model flags a fault) — it's not a separate manually-maintained
  database, it grows from the system's own operational history.
- The RAG corpus is grounded in the project's own domain knowledge
  (the UCI dataset's failure-mode taxonomy: pump, valve, accumulator,
  cooler) rather than generic text, so retrieval quality is high
  even with a small, curated document set.
- The LLM is deliberately used only for tool orchestration and
  synthesis, not for the actual fault detection — that keeps the
  chatbot's answers grounded in real model output and real documents
  instead of the LLM guessing at sensor physics.
