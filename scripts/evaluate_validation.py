import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.validation.agent import build_graph
from src.eval.harness import load_golden_dataset, run_validation_case
from src.eval.metrics import validation_factual


def main():
    """Evaluate the validation agent on the golden dataset (model-free)."""
    dataset = load_golden_dataset()
    app = build_graph()
    correct = 0
    for case in dataset:
        verdict = run_validation_case(app, case)
        m = validation_factual(verdict, case["expected"]["validation"])
        correct += m["status_correct"]
        print(f"{case['id']}: status={m['status']} "
              f"expected={case['expected']['validation']['status']} "
              f"status_ok={m['status_correct']} coverage_ok={m['coverage_correct']}")
    print(f"\nValidation accuracy: {correct}/{len(dataset)}")


if __name__ == "__main__":
    main()
