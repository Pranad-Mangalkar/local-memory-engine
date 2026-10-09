import os
from engine import MemoryEngine


def main():
    db_file = "engine_demo.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    engine = MemoryEngine(db_path=db_file)

    print("--- 1. Testing Ingestion & Small Talk Filter ---")
    res1 = engine.process_message("ok thanks")
    print(f"Message: 'ok thanks' -> Result: {res1} (Expected: None)")

    res2 = engine.process_message("I use Arch Linux.")
    print(f"Message: 'I use Arch Linux.' -> Stored ID: {res2['id']}")

    res3 = engine.process_message("My project deadline is October 15.")
    print(f"Message: 'deadline is October 15.' -> Stored ID: {res3['id']}")

    print("\n--- 2. Testing Update / Replacement ---")
    res4 = engine.process_message("Actually, my project deadline is November 2.")
    print(
        f"Message: 'deadline is November 2.' -> Stored ID: {res4['id']}, Replaced ID: {res4['replaced_memory_id']}"
    )

    print("\n--- 3. Testing Semantic Recall ---")
    r_os = engine.recall("What OS does the user use?")
    print(
        f"Query: 'What OS does the user use?' -> {[m['text'] for m in r_os]}"
    )

    r_deadline = engine.recall("When is the project deadline?")
    print(
        f"Query: 'When is the project deadline?' -> {[m['text'] for m in r_deadline]}"
    )

    r_unknown = engine.recall("What is the user's favourite food?")
    print(
        f"Query: 'What is the user's favourite food?' -> {r_unknown} (Expected: [])"
    )


if __name__ == "__main__":
    main()