import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.declaration.agent import build_graph as build_declaration_graph
from src.agents.declaration.inference import HFInference
from src.agents.validation.agent import build_graph as build_validation_graph
from src.agents.orchestrator.agent import select_provider
from src.eval.harness import load_golden_dataset, run_declaration_case, run_validation_case
from src.eval.metrics import (
    declaration_completeness,
    validation_factual,
    orchestration_provider,
)
from src.eval.report import build_row, aggregates, write_csv

CSV_PATH = Path(__file__).parent.parent / "data" / "evaluation_results.csv"


def main():
    """Run all three agent evaluations and consolidate per-case metrics into a CSV (needs GPU)."""
    dataset = load_golden_dataset()
    decl_app = build_declaration_graph(HFInference())
    val_app = build_validation_graph()

    rows = []
    for case in dataset:
        extracted = run_declaration_case(decl_app, case)
        decl_m = declaration_completeness(extracted)
        verdict = run_validation_case(val_app, case)
        val_m = validation_factual(verdict, case["expected"]["validation"])
        chosen = select_provider(case["expected"]["declaration"]["incident_type"])
        orch_m = orchestration_provider(chosen, case["expected"]["orchestration"]["provider"])
        rows.append(build_row(case, decl_m, val_m, orch_m))

    write_csv(rows, CSV_PATH)
    print(f"Wrote {CSV_PATH}")
    for k, v in aggregates(rows).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
