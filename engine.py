import re
from typing import Any, Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from storage import MemoryStore


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


TRIVIAL_PATTERNS = [
    r"^(ok|okay|thanks|thank you|cool|great|hello|hi|hey|bye|goodbye)[\.!\s]*$",
    r"^(yes|no|yep|nope|sure|got it|understood)[\.!\s]*$",
]


class MemoryEngine:

    def __init__(
        self,
        db_path: str = "memory_engine.db",
        model_name: str = "all-MiniLM-L6-v2",
        relevance_threshold: float = 0.55,
        conflict_threshold: float = 0.70,
    ):
        self.store = MemoryStore(db_path)
        self.model = SentenceTransformer(model_name)
        self.relevance_threshold = relevance_threshold
        self.conflict_threshold = conflict_threshold

    def is_small_talk(self, message: str) -> bool:
        clean = message.strip().lower()
        if len(clean) < 3:
            return True
        for pattern in TRIVIAL_PATTERNS:
            if re.match(pattern, clean):
                return True
        return False

    def extract_fact_details(self, message: str) -> Optional[Dict[str, str]]:
        clean = message.strip()
        lower = clean.lower()

        if any(w in lower for w in ["deadline", "project", "building", "app"]):
            mem_type = "project"
        elif any(
            w in lower for w in ["prefer", "favourite", "favorite", "like", "use"]
        ):
            mem_type = "preference"
        elif any(w in lower for w in ["goal", "target", "want to", "aim"]):
            mem_type = "goal"
        elif any(
            w in lower
            for w in [
                "meeting",
                "tomorrow",
                "event",
                "conference",
                "scheduled",
            ]
        ):
            mem_type = "event"
        else:
            mem_type = "fact"

        return {"text": clean, "type": mem_type}

    def process_message(self, message: str) -> Optional[Dict[str, Any]]:
        if self.is_small_talk(message):
            return None

        extracted = self.extract_fact_details(message)
        if not extracted:
            return None

        new_text = extracted["text"]
        new_type = extracted["type"]
        new_vec = self.model.encode(new_text)

        active_memories = self.store.get_active_memories()
        conflicting_id = None

        for memory in active_memories:
            sim = cosine_similarity(new_vec, memory["embedding"])
            if sim >= self.conflict_threshold:
                conflicting_id = memory["id"]
                break

        new_id = self.store.insert_memory(
            text=new_text,
            memory_type=new_type,
            source_message=message,
            embedding=new_vec,
        )

        if conflicting_id is not None:
            self.store.mark_replaced(
                old_memory_id=conflicting_id, new_memory_id=new_id
            )

        return {
            "id": new_id,
            "text": new_text,
            "type": new_type,
            "replaced_memory_id": conflicting_id,
        }

    def recall(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        active_memories = self.store.get_active_memories()
        if not active_memories:
            return []

        query_vec = self.model.encode(query)
        scored = []

        for mem in active_memories:
            score = cosine_similarity(query_vec, mem["embedding"])
            if score >= self.relevance_threshold:
                scored.append({
                    "id": mem["id"],
                    "text": mem["text"],
                    "type": mem["type"],
                    "score": round(float(score), 4),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]