from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine import MemoryEngine

app = FastAPI(
    title="Local Memory Engine",
    description="Local memory extraction, persistence, and semantic retrieval engine.",
    version="1.0.0",
)

engine = MemoryEngine(db_path="memory_engine.db")


class MessageRequest(BaseModel):
    message: str = Field(..., example="I use Arch Linux.")


class RecallRequest(BaseModel):
    query: str = Field(..., example="What OS does the user use?")
    top_k: int = Field(5, ge=1, le=20, example=5)


class RecallResponse(BaseModel):
    memories: List[Dict[str, Any]]


@app.post("/memory")
def add_memory(req: MessageRequest):
    result = engine.process_message(req.message)
    if result is None:
        return {
            "status": "ignored",
            "reason": "Filtered as small talk or uninformative message.",
        }
    return {"status": "saved", "memory": result}


@app.post("/recall", response_model=RecallResponse)
def recall_memories(req: RecallRequest):
    results = engine.recall(query=req.query, top_k=req.top_k)
    return {"memories": results}


@app.get("/memories")
def list_memories():
    all_memories = engine.store.get_all_memories()
    return {"total": len(all_memories), "memories": all_memories}


@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: int):
    success = engine.store.delete_memory(memory_id)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Memory ID {memory_id} not found."
        )
    return {"status": "deleted", "id": memory_id}