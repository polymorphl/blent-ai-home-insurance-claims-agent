import json
from pathlib import Path

from langchain_core.messages import HumanMessage

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_dataset_full.json"


def load_golden_dataset(path: Path | None = None) -> list[dict]:
    """Load the unified golden dataset."""
    return json.loads((path or DATASET_PATH).read_text())


def run_declaration_case(app, case: dict) -> dict:
    """Drive the declaration agent over case['turns']; return the extracted claim fields.

    Returns the completed final_claim when the agent finishes, otherwise the partial
    claim_data collected so far (so completeness reflects partial extraction).
    """
    config = {"configurable": {"thread_id": f"eval_{case['id']}"}}
    first = True
    result = None
    for turn in case["turns"]:
        if first:
            state = {
                "messages": [HumanMessage(content=turn["content"])],
                "claim_data": {}, "missing_fields": [],
                "is_complete": False, "final_claim": None,
            }
            first = False
        else:
            state = {"messages": [HumanMessage(content=turn["content"])]}
        result = app.invoke(state, config)
        if result.get("is_complete"):
            break
    result = result or {}
    return result.get("final_claim") or result.get("claim_data") or {}


def run_validation_case(app, case: dict) -> dict:
    """Invoke the validation agent on the gold claim + today_override; return the verdict."""
    claim = case["expected"]["declaration"]
    result = app.invoke({
        "claim": claim,
        "conformity_errors": [], "coverage_errors": [], "photo_errors": [],
        "verdict": None, "today_override": case.get("today_override"),
    })
    return result["verdict"]
