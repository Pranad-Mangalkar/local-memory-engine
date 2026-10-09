import numpy as np
from storage import MemoryStore


def main():
    store = MemoryStore("test_engine.db")

    dummy_vec_1 = np.ones(384, dtype=np.float32)
    dummy_vec_2 = np.zeros(384, dtype=np.float32)

    id1 = store.insert_memory(
        text="User's deadline is October 15.",
        memory_type="project",
        source_message="My project deadline is October 15.",
        embedding=dummy_vec_1,
    )
    print(f"Inserted Memory 1 with ID: {id1}")

    id2 = store.insert_memory(
        text="User's deadline is November 2.",
        memory_type="project",
        source_message="Actually, the deadline moved to November 2.",
        embedding=dummy_vec_2,
    )
    store.mark_replaced(old_memory_id=id1, new_memory_id=id2)
    print(f"Inserted Memory 2 with ID: {id2} and marked {id1} as replaced.")

    active = store.get_active_memories()
    print(f"\nActive memories count: {len(active)} (Expected: 1)")
    print(f"Active memory text: '{active[0]['text']}'")

    all_records = store.get_all_memories()
    print(f"\nTotal audit records: {len(all_records)} (Expected: 2)")
    for rec in all_records:
        print(
            f"  ID {rec['id']}: '{rec['text']}' | active={rec['is_active']} | replaced_by={rec['replaced_by_id']}"
        )


if __name__ == "__main__":
    main()