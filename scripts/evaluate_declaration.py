import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.declaration.agent import build_graph
from src.agents.declaration.inference import HFInference
from src.eval.harness import load_golden_dataset, run_declaration_case
from src.eval.metrics import declaration_completeness


def main():
    """Evaluate the declaration agent's field completeness on the golden dataset (needs GPU)."""
    dataset = load_golden_dataset()
    app = build_graph(HFInference())
    total = 0.0
    for case in dataset:
        extracted = run_declaration_case(app, case)
        m = declaration_completeness(extracted)
        total += m["score"]
        print(f"{case['id']}: completeness={m['score']:.2f} fields={m['fields_ok']}")
    print(f"\nMean completeness: {total / len(dataset):.2f}")


if __name__ == "__main__":
    main()
