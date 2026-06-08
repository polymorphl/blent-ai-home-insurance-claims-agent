import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator.agent import select_provider
from src.eval.harness import load_golden_dataset
from src.eval.metrics import orchestration_provider


def main():
    """Evaluate orchestration provider selection on the golden dataset (model-free)."""
    dataset = load_golden_dataset()
    correct = 0
    for case in dataset:
        chosen = select_provider(case["expected"]["declaration"]["incident_type"])
        expected = case["expected"]["orchestration"]["provider"]
        m = orchestration_provider(chosen, expected)
        correct += m["correct"]
        mark = "OK" if m["correct"] else "X"
        print(f"{case['id']}: provider={m['provider']} expected={expected} [{mark}]")
    print(f"\nProvider precision: {correct}/{len(dataset)}")


if __name__ == "__main__":
    main()
