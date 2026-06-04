from datetime import date

from langgraph.graph import END, StateGraph

from src.agents.validation.rules import COVERAGE_RULES, check_conformity, check_coverage
from src.agents.validation.state import ValidationState


def check_conformity_node(state: ValidationState) -> dict:
    """Check that all required claim fields are present and photos are provided."""
    return {"conformity_errors": check_conformity(state["claim"])}


def check_coverage_node(state: ValidationState) -> dict:
    """Check that the incident type is covered and the declaration deadline is respected."""
    today_str = state.get("today_override")
    today = date.fromisoformat(today_str) if today_str else None
    return {"coverage_errors": check_coverage(state["claim"], today=today)}


def finalize_node(state: ValidationState) -> dict:
    """Build the final verdict from accumulated conformity and coverage errors."""
    all_errors = state.get("conformity_errors", []) + state.get("coverage_errors", [])
    claim = state["claim"]
    if all_errors:
        return {"verdict": {
            "status": "rejected",
            "reason": all_errors[0],
            "coverage": None,
            "claim": claim,
        }}
    rule = COVERAGE_RULES.get(claim["incident_type"], {})
    return {"verdict": {
        "status": "approved",
        "reason": "Dossier validé.",
        "coverage": {"ceiling": rule["ceiling"], "deductible": rule["deductible"]} if rule else None,
        "claim": claim,
    }}


def build_graph():
    """Build the validation workflow graph."""
    workflow = StateGraph(ValidationState)

    workflow.add_node("check_conformity", check_conformity_node)
    workflow.add_node("check_coverage", check_coverage_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("check_conformity")
    workflow.add_conditional_edges(
        "check_conformity",
        lambda s: "finalize" if s.get("conformity_errors") else "check_coverage",
    )
    workflow.add_edge("check_coverage", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
