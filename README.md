# Local Memory Engine for AI Chatbots

A lightweight, local memory extraction, persistence, and semantic recall engine designed for conversational AI agents. Built completely offline on consumer hardware with zero paid or cloud API dependencies.

---

## 1. Architecture Overview

```
                      +-------------------------+
                      |       Chat Client       |
                      |   (REST API / Swagger)  |
                      +------------+------------+
                                   | HTTP
                                   v
                      +-------------------------+
                      |   main.py (FastAPI)     |
                      +------------+------------+
                                   |
                                   v
                      +-------------------------+
                      |   engine.py (Logic)     |
                      |  - Small-talk filtering |
                      |  - Subject extraction   |
                      |  - Contradiction check  |
                      |  - Relevance gating     |
                      +------+------------+-----+
                             |            |
                 Vector Ops  v            v  Relational CRUD
              +------------------+    +-----------------------+
              | sentence-transf. |    |  storage.py (SQLite)  |
              | all-MiniLM-L6-v2 |    |  memories + audit log |
              +------------------+    +-----------------------+
```

### Module Responsibilities
* **`storage.py` (Persistence Layer):** Manages SQLite connections, schema initialization, binary BLOB serialization for dense vector arrays, soft-delete replacement state updates (`is_active`, `replaced_by_id`), and raw CRUD operations.
* **`engine.py` (Memory Logic & Retrieval):** Orchestrates noise/small-talk rejection, subject domain classification, local embedding calculation via `sentence-transformers`, topic-level contradiction detection, and threshold-gated cosine similarity recall.
* **`main.py` (API Layer):** Exposes four standard REST endpoints (`POST /memory`, `POST /recall`, `GET /memories`, `DELETE /memory/{id}`) using FastAPI with Pydantic schema validation.
* **`evaluate.py` (Benchmarking Suite):** Automated test suite running sequential conversation messages and recall queries against an isolated SQLite test database.

---

## 2. Setup and Execution

### Prerequisites
* Python 3.10+
* Virtual environment (`venv`)

### Installation
```powershell
# Clone the repository
git clone https://github.com/Pranad-Mangalkar/local-memory-engine.git
cd local-memory-engine

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install fastapi uvicorn pydantic sentence-transformers numpy
```

### Running the API Server
```powershell
uvicorn main:app --reload
```
Access the interactive Swagger UI documentation at: `http://127.0.0.1:8000/docs`

### Running the Benchmark Suite
```powershell
python evaluate.py
```

---

## 3. Evaluation Results

The evaluation suite validates the engine across 33 total test interactions (16 sequential conversation messages and 17 comprehensive recall queries):
* **Direct Facts & Preferences:** Accurately extracts and indexes facts (e.g., Python, PostgreSQL, Arch Linux).
* **Superseded Information / Contradictions:** Verifies that when a deadline moves from October 15 to November 2, only November 2 is retrieved while preserving October 15 in the historical audit log.
* **Multi-Memory / Semantic Associations:** Verifies matching of semantic context over exact keyword repetition.
* **Negative / Ungrounded Queries:** Tests 5 out-of-distribution queries (food, pets, movies, flights, music); strictly returns empty results (`[]`) rather than hallucinating low-confidence memories.

**Overall Benchmark Accuracy:** **94.12%** (16/17 passed)

---

## 4. Technical Design Questions

### Why did you choose your storage and retrieval setup?
* **Storage (SQLite):** Embedded, serverless, and requires zero daemon setup. Supports exact relational queries (`is_active = 1`) alongside serialized binary `BLOB` storage of raw `float32` vector arrays.
* **Retrieval (`all-MiniLM-L6-v2` via SentenceTransformers):** Computes 384-dimensional dense semantic vectors with fast CPU inference latency (<15ms per query), satisfying the local-first execution requirement without external API calls or high RAM footprints.

### How do you decide what becomes a memory?
Incoming messages pass through a noise filter that strips punctuation and compares tokens against conversational filler sets (`TRIVIAL_WORDS`). Messages consisting entirely of conversational acknowledgments (e.g., "ok thanks for the tip", "cool got it") are discarded immediately, preventing junk writes. Informative messages are classified into structured categories (`fact`, `preference`, `project`, `event`, `goal`) along with a subject key.

### How do you handle updated or contradictory information?
When an informative message arrives, the engine performs topic and domain matching against active records in SQLite. If a match exceeds the contradiction similarity threshold ($\ge 0.55$), the engine marks the older record as inactive (`is_active = 0`) and populates `replaced_by_id` with the new memory ID. The superseded fact is excluded from future `/recall` operations while remaining preserved for audit tracking.

### What happens when the answer is not in memory?
Retrieval enforces a calibrated similarity threshold ($\ge 0.30$). If a query's cosine similarity against all active memories fails to reach this threshold, the engine returns an empty list (`[]`). This prevents downstream conversational LLMs from hallucinating answers based on unrelated facts.

### What would break first with 10 million memories, and how would you fix it?
1. **The Bottleneck:** At 10 million memories, brute-force linear vector scan ($O(N)$ across 384-dimensional embeddings) will cause query latency to degrade to several seconds per call, and deserializing 10 million BLOBs will exhaust working memory.
2. **The Fix:**
   * Transition from brute-force cosine comparison to an Approximate Nearest Neighbor (ANN) index such as **HNSW** or **IVFFlat** via FAISS, USearch, or SQLite vector extensions like `sqlite-vec`.
   * Apply metadata partitioning: filter vectors first using SQL indexes on `user_id` or `subject_domain`, running vector search only over the relevant subset ($N < 10,000$).
   * Store quantized embeddings (INT8 or scalar quantization) to reduce memory bandwidth by 75%.