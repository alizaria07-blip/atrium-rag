"""A tiny mock of an OpenAI-compatible /v1 API, for local testing.

Implements: GET /v1/models, POST /v1/embeddings, POST /v1/chat/completions
(streaming and non-streaming). Embeddings are deterministic char-trigram
hashing so that texts sharing words get similar vectors — enough to exercise
the RAG search. When the chat system prompt asks for planner JSON, it returns
a plausible read-only SQL statement so the warehouse path can be tested.
"""
import json
import hashlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
DIM = 256


def embed_text(text: str) -> list[float]:
    vec = [0.0] * DIM
    t = text.lower()
    for i in range(len(t) - 2):
        gram = t[i:i + 3]
        h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def _planner_sql(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ("revenue", "sales", "total")):
        return (
            "SELECT p.category, SUM(oi.qty * oi.unit_price) AS revenue "
            "FROM order_items oi JOIN products p ON p.id = oi.product_id "
            "GROUP BY p.category ORDER BY revenue DESC LIMIT 5"
        )
    return "SELECT status, COUNT(*) AS orders FROM orders GROUP BY status"


def _planner_reply(body: dict) -> JSONResponse:
    user = next((m["content"] for m in body.get("messages", []) if m["role"] == "user"), "")
    payload = {
        "id": "chatcmpl-mock-planner",
        "object": "chat.completion",
        "created": 1,
        "model": body.get("model", "mock-chat"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"sql": _planner_sql(user), "reason": "Planned warehouse query (mock)"}
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }
    return JSONResponse(payload)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "mock-chat", "object": "model"},
                                       {"id": "mock-embed", "object": "model"}]}


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    inputs = body["input"]
    if isinstance(inputs, str):
        inputs = [inputs]
    data = [{"object": "embedding", "index": i, "embedding": embed_text(x)}
            for i, x in enumerate(inputs)]
    return {"object": "list", "data": data, "model": body.get("model", "")}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body["messages"]
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    if "Return ONLY JSON" in system:
        return _planner_reply(body)
    if not body.get("stream"):
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model", "mock-chat"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Mock answer. Question: {user[:40]}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        return JSONResponse(payload)
    ctx = ""
    if "CONTEXT:" in system:
        ctx = system.split("CONTEXT:", 1)[1].strip()
    sources = [ln.split("] ")[0].strip("[]") + ") " for ln in ctx.splitlines() if ln.startswith("[")]
    reply = f"Mock answer. Question: {user[:40]}... Found {len(sources)} source chunk(s)."
    tokens = [reply[i:i + 12] for i in range(0, len(reply), 12)]

    def gen():
        yield _chunk({"id": "c", "object": "chat.completion.chunk",
                      "created": 1, "model": body.get("model"),
                      "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        for tok in tokens:
            yield _chunk({"id": "c", "object": "chat.completion.chunk", "created": 1,
                          "model": body.get("model"),
                          "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]})
        yield _chunk({"id": "c", "object": "chat.completion.chunk", "created": 1,
                      "model": body.get("model"),
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _chunk(obj):
    return f"data: {json.dumps(obj)}\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=59998)