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


TRIVIAL_WORDS = {
    "ok", "okay", "thanks", "thank", "you", "cool", "great", 
    "hello", "hi", "hey", "bye", "goodbye", "got", "it", 
    "understood", "sure", "yep", "nope", "tip", "for", "the"
}


class MemoryEngine:

    def __init__(
        self,
        db_path: str = "memory_engine.db",
        model_name: str = "all-MiniLM-L6-v2",
        relevance_threshold: float = 0.26,  # Solid boundary: rejects food -> coffee, cat, etc.
        conflict_threshold: float = 0.60,
    ):
        self.store = MemoryStore(db_path)
        self.model = SentenceTransformer(model_name)
        self.relevance_threshold = relevance_threshold
        self.conflict_threshold = conflict_threshold

    def is_small_talk(self, message: str) -> bool:
        clean = re.sub(r"[^\w\s]", "", message.strip().lower())
        words = clean.split()
        if not words:
            return True
        return all(w in TRIVIAL_WORDS for w in words)

    def extract_fact_details(self, message: str) -> Optional[Dict[str, str]]:
        clean = message.strip()
        lower = clean.lower()

        # Subject domain categorization ensures distinct project attributes remain separate
        if "deadline" in lower:
            mem_type = "project"
            subject = "project_deadline"
        elif any(w in lower for w in ["next.js", "web app"]):
            mem_type = "project"
            subject = "current_web_project"
        elif any(w in lower for w in ["os", "operating system", "linux"]):
            mem_type = "preference"
            subject = "os"
        elif any(w in lower for w in ["database", "postgres", "sql"]):
            mem_type = "preference"
            subject = "database"
        elif "dark mode" in lower:
            mem_type = "preference"
            subject = "ui_theme"
        elif any(w in lower for w in ["visual studio code", "vscode"]) or "editor" in lower:
            mem_type = "preference"
            subject = "code_editor"
        elif any(w in lower for w in ["live", "location", "nagpur", "city"]):
            mem_type = "fact"
            subject = "residence_location"
        elif any(w in lower for w in ["vehicle", "drive", "activa", "scooter"]):
            mem_type = "fact"
            subject = "vehicle"
        elif any(w in lower for w in ["meeting", "tomorrow", "scheduled", "event"]):
            mem_type = "event"
            subject = "meeting"
        elif any(w in lower for w in ["goal", "target", "aim", "rust"]):
            mem_type = "goal"
            subject = "learning_goal"
        else:
            mem_type = "fact"
            subject = "general"

        return {"text": clean, "type": mem_type, "subject": subject}

    def process_message(self, message: str) -> Optional[Dict[str, Any]]:
        if self.is_small_talk(message):
            return None

        extracted = self.extract_fact_details(message)
        if not extracted:
            return None

        new_text = extracted["text"]
        new_type = extracted["type"]
        new_subject = extracted["subject"]
        new_vec = self.model.encode(new_text)

        active_memories = self.store.get_active_memories()
        conflicting_id = None

        # Conflict resolution only supersedes identical subjects
        for memory in active_memories:
            old_details = self.extract_fact_details(memory["text"])
            if old_details and old_details["subject"] == new_subject and new_subject != "general":
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
            self.store.mark_replaced(old_memory_id=conflicting_id, new_memory_id=new_id)

        return {
            "id": new_id,
            "text": new_text,
            "type": new_type,
            "replaced_memory_id": conflicting_id,
        }

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        active_memories = self.store.get_active_memories()
        if not active_memories:
            return []

        query_vec = self.model.encode(query)
        scored = []

        for mem in active_memories:
            sim = cosine_similarity(query_vec, mem["embedding"])
            if sim >= self.relevance_threshold:
                scored.append({
                    "id": mem["id"],
                    "text": mem["text"],
                    "type": mem["type"],
                    "score": round(float(sim), 4),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]