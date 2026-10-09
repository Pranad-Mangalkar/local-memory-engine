import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    stored_memories = [
        "User operates on Arch Linux.",
        "User's project deadline is November 2.",
        "User is building a Next.js application.",
    ]

    test_queries = [
        ("What OS does the user use?", "Should match: Arch Linux"),
        ("When is the project deadline?", "Should match: deadline"),
        ("What is the user's favourite food?", "Should return NOTHING"),
        ("Does the user have any pets?", "Should return NOTHING"),
    ]

    print("\n--- Encoding Memories ---")
    memory_embeddings = model.encode(stored_memories)

    print("\n--- Running Similarity Tests ---")
    for query, expected_behavior in test_queries:
        query_vec = model.encode(query)
        scores = [
            (mem, cosine_similarity(query_vec, mem_vec))
            for mem, mem_vec in zip(stored_memories, memory_embeddings)
        ]

        scores.sort(key=lambda x: x[1], reverse=True)
        top_mem, top_score = scores[0]

        print(f"\nQuery: '{query}' ({expected_behavior})")
        print(f"  Top Match: '{top_mem}'")
        print(f"  Score: {top_score:.4f}")


if __name__ == "__main__":
    main()