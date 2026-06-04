from datetime import date
from pathlib import Path

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


def make_check_photos_node(vlm_inference, attachments_dir: Path):
    """Factory for the photo coherence check node."""
    def check_photos_node(state: ValidationState) -> dict:
        """Verify that attached images match the declared incident type (majority rule)."""
        if vlm_inference is None:
            return {"photo_errors": []}

        claim = state["claim"]
        incident_type = claim.get("incident_type")
        photo_filenames = claim.get("photo_filenames", [])

        if not photo_filenames:
            return {"photo_errors": []}

        results = []
        for filename in photo_filenames:
            image_path = attachments_dir / filename
            if not image_path.exists():
                continue
            coherent = vlm_inference.check_photo_coherence(str(image_path), incident_type)
            results.append(coherent)

        if not results:
            return {"photo_errors": []}

        matching = sum(1 for r in results if r)
        if matching / len(results) <= 0.5:
            return {"photo_errors": [
                f"Les photos ne correspondent pas au type de sinistre déclaré ({incident_type})."
            ]}
        return {"photo_errors": []}

    return check_photos_node


def finalize_node(state: ValidationState) -> dict:
    """Build the final verdict from accumulated conformity, coverage and photo errors."""
    all_errors = (
        state.get("conformity_errors", [])
        + state.get("coverage_errors", [])
        + state.get("photo_errors", [])
    )
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


def build_graph(vlm_inference=None, attachments_dir: Path | None = None):
    """Build the validation workflow graph."""
    if attachments_dir is None:
        attachments_dir = Path("data/attachments")

    workflow = StateGraph(ValidationState)

    workflow.add_node("check_conformity", check_conformity_node)
    workflow.add_node("check_coverage", check_coverage_node)
    workflow.add_node("check_photos", make_check_photos_node(vlm_inference, attachments_dir))
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("check_conformity")
    workflow.add_conditional_edges(
        "check_conformity",
        lambda s: "finalize" if s.get("conformity_errors") else "check_coverage",
    )
    workflow.add_conditional_edges(
        "check_coverage",
        lambda s: "finalize" if s.get("coverage_errors") else "check_photos",
    )
    workflow.add_edge("check_photos", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
