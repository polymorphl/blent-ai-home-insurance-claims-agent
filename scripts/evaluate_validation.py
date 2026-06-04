import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.validation.agent import build_graph


def evaluate_example(result: dict, expected: dict) -> dict:
    """Compare verdict status against expected value."""
    verdict = result.get("verdict") or {}
    status = verdict.get("status")
    return {
        "status_correct": status == expected.get("status"),
        "status": status,
        "expected_status": expected.get("status"),
        "reason": verdict.get("reason"),
        "coverage": verdict.get("coverage"),
    }


def main():
    """Load the validation dataset, run all cases through the agent, and compute accuracy."""
    dataset_path = Path(__file__).parent.parent / "data" / "golden_dataset_validation.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    app = build_graph()

    all_results = []
    for example in dataset:
        print(f"\nRunning: {example['id']}")
        result = app.invoke({
            "claim": example["claim"],
            "conformity_errors": [],
            "coverage_errors": [],
            "verdict": None,
            "today_override": example.get("today_override"),
        })
        metrics = evaluate_example(result, example["expected"])
        all_results.append({"id": example["id"], **metrics})
        print(f"  Status correct: {metrics['status_correct']}")
        print(f"  Got: {metrics['status']}, Expected: {metrics['expected_status']}")
        print(f"  Reason: {metrics['reason']}")

    acc = sum(r["status_correct"] for r in all_results) / len(all_results) if all_results else 0.0
    print(f"\n{'=' * 40}")
    print(f"Accuracy: {acc:.2f} ({sum(r['status_correct'] for r in all_results)}/{len(all_results)})")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
