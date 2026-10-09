import json
import os
from engine import MemoryEngine


def run_evaluation(dataset_path: str = "eval_dataset.json"):
    test_db = "eval_test.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    engine = MemoryEngine(db_path=test_db)

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    messages = [c for c in cases if c["type"] == "message"]
    queries = [c for c in cases if c["type"] == "query"]

    print(f"--- Ingesting {len(messages)} sample messages ---")
    for item in messages:
        res = engine.process_message(item["text"])
        if res:
            print(f"  [SAVED] {res['text']} (type={res['type']})")
        else:
            print(f"  [IGNORED] '{item['text']}'")

    print(f"\n--- Evaluating {len(queries)} Recall Queries ---")
    correct = 0
    total = len(queries)

    for case in queries:
        q = case["query"]
        should_find = case["should_find"]
        exp_kw = case.get("expected_keywords", [])
        unexp_kw = case.get("unexpected_keywords", [])

        results = engine.recall(q, top_k=3)

        passed = False
        if not should_find:
            if len(results) == 0:
                passed = True
        else:
            if len(results) > 0:
                top_text = results[0]["text"]
                has_exp = all(kw.lower() in top_text.lower() for kw in exp_kw)
                has_no_unexp = not any(
                    kw.lower() in top_text.lower() for kw in unexp_kw
                )
                if has_exp and has_no_unexp:
                    passed = True

        status = "PASS" if passed else "FAIL"
        if passed:
            correct += 1

        print(f"[{status}] Query: '{q}'")
        if not passed:
            print(f"   Returned: {[r['text'] for r in results]}")

    accuracy = (correct / total) * 100
    print("\n===============================")
    print(f"Total Queries: {total}")
    print(f"Passed: {correct}/{total}")
    print(f"Benchmark Accuracy: {accuracy:.2f}%")
    print("===============================")


if __name__ == "__main__":
    run_evaluation()