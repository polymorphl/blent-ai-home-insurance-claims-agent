import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.agents.declaration.agent import build_graph
from src.agents.declaration.inference import HFInference
from src.agents.declaration.tools import REQUIRED_FIELDS, completeness_score


def evaluate_example(result: dict, expected: dict) -> dict:
    """Compare extracted claim fields against expected values and compute completeness metrics."""
    final = result.get("final_claim") or {}
    per_field = {f: final.get(f) is not None for f in REQUIRED_FIELDS}
    return {
        "score": completeness_score(final),
        "turns_count": final.get("turns_count", 0),
        "per_field": per_field,
        "extracted": {f: final.get(f) for f in REQUIRED_FIELDS},
        "expected": {f: expected.get(f) for f in REQUIRED_FIELDS},
    }


def run_dataset_example(app, example: dict) -> dict:
    """Execute a dataset example through the claim agent until completion or all turns exhausted."""
    config = {"configurable": {"thread_id": example["id"]}}
    first_turn = True
    result = None

    for turn in example["turns"]:
        if first_turn:
            state = {
                "messages": [HumanMessage(content=turn["content"])],
                "claim_data": {},
                "missing_fields": [],
                "is_complete": False,
                "final_claim": None,
            }
            first_turn = False
        else:
            state = {"messages": [HumanMessage(content=turn["content"])]}

        result = app.invoke(state, config)
        if result["is_complete"]:
            break

    return result


def main():
    """Load the dataset, run all examples through the claim agent, and compute average completeness metrics."""
    dataset_path = Path(__file__).parent.parent / "data" / "golden_dataset.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    print("Loading model...")
    inference = HFInference()
    app = build_graph(inference)

    all_results = []
    for example in dataset:
        print(f"\nRunning: {example['id']}")
        run_result = run_dataset_example(app, example)
        metrics = evaluate_example(run_result, example["expected"])
        all_results.append({"id": example["id"], **metrics})
        print(f"  Score:     {metrics['score']:.2f}")
        print(f"  Turns:     {metrics['turns_count']}")
        print(f"  Per field: {metrics['per_field']}")

    avg = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
    print(f"\n{'=' * 40}")
    print(f"Average completeness score: {avg:.2f}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
