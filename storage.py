from datetime import datetime, timezone
import sqlite3
from typing import Any, Dict, List, Optional
import numpy as np


class MemoryStore:

    def __init__(self, db_path: str = "memory_engine.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source_message TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    timestamp TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    replaced_by_id INTEGER DEFAULT NULL,
                    FOREIGN KEY (replaced_by_id) REFERENCES memories(id)
                );
            """)
            conn.commit()

    @staticmethod
    def _embedding_to_blob(embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def _blob_to_embedding(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def insert_memory(
        self,
        text: str,
        memory_type: str,
        source_message: str,
        embedding: np.ndarray,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        blob = self._embedding_to_blob(embedding)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories (text, type, source_message, embedding, timestamp, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
            """,
                (text, memory_type, source_message, blob, now),
            )
            conn.commit()
            return cursor.lastrowid

    def mark_replaced(self, old_memory_id: int, new_memory_id: int):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories
                SET is_active = 0, replaced_by_id = ?
                WHERE id = ?
            """,
                (new_memory_id, old_memory_id),
            )
            conn.commit()

    def get_active_memories(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, text, type, source_message, timestamp, embedding
                FROM memories
                WHERE is_active = 1
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()

        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "text": r["text"],
                "type": r["type"],
                "source_message": r["source_message"],
                "timestamp": r["timestamp"],
                "embedding": self._blob_to_embedding(r["embedding"]),
            })
        return results

    def get_all_memories(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, text, type, source_message, timestamp, is_active, replaced_by_id
                FROM memories
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def delete_memory(self, memory_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0