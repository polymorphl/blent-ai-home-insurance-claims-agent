from collections import Counter
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.agents.expertise.prompts import EXPERTISE_SUMMARY_PROMPT
from src.agents.expertise.rules import estimate_costs
from src.agents.expertise.state import ExpertiseState


def make_assess_severity_node(vlm_inference, attachments_dir: Path):
    """Factory for the damage severity assessment node."""
    def assess_severity_node(state: ExpertiseState) -> dict:
        """Assess damage severity from attached photos using VLM (majority rule)."""
        claim = state["verdict"]["claim"]
        photo_filenames = claim.get("photo_filenames", [])

        if vlm_inference is None or not photo_filenames:
            return {"severity": "unknown"}

        results = []
        for filename in photo_filenames:
            image_path = attachments_dir / filename
            if not image_path.exists():
                continue
            severity = vlm_inference.assess_damage_severity(
                str(image_path), claim.get("incident_type")
            )
            results.append(severity)

        if not results:
            return {"severity": "unknown"}

        counts = Counter(results)
        max_count = counts.most_common(1)[0][1]
        tied = [s for s, c in counts.items() if c == max_count]
        severity_order = ["low", "medium", "high"]
        winner = max(tied, key=lambda s: severity_order.index(s) if s in severity_order else -1)
        return {"severity": winner}

    return assess_severity_node


def estimate_costs_node(state: ExpertiseState) -> dict:
    """Estimate costs based on coverage and severity."""
    verdict = state["verdict"]
    coverage = verdict.get("coverage")
    if coverage is None:
        return {"cost_range": (0, 0), "compensable_amount": (0, 0),
                "deductible_applied": 0, "ceiling_applied": 0}
    claim = verdict["claim"]
    return estimate_costs(
        incident_type=claim["incident_type"],
        severity=state.get("severity") or "unknown",
        ceiling=coverage["ceiling"],
        deductible=coverage["deductible"],
    )


def make_generate_report_node(inference):
    """Factory for the report generation node."""
    def generate_report_node(state: ExpertiseState) -> dict:
        """Generate expertise summary using LLM inference."""
        if inference is None:
            return {"summary": "Rapport d'expertise non disponible (modèle absent)."}

        claim = state["verdict"]["claim"]
        cost_range = state.get("cost_range") or (0, 0)
        comp_amount = state.get("compensable_amount") or (0, 0)

        prompt = EXPERTISE_SUMMARY_PROMPT.format(
            incident_type=claim.get("incident_type", ""),
            description=claim.get("description", ""),
            date=claim.get("date", ""),
            severity=state.get("severity") or "unknown",
            cost_low=cost_range[0],
            cost_high=cost_range[1],
            comp_low=comp_amount[0],
            comp_high=comp_amount[1],
            deductible=state.get("deductible_applied") or 0,
            ceiling=state.get("ceiling_applied") or 0,
        )
        messages = [{"role": "user", "content": prompt}]
        return {"summary": inference.generate(messages)}

    return generate_report_node


def finalize_node(state: ExpertiseState) -> dict:
    """Build the final expertise report from accumulated analysis."""
    return {"report": {
        "severity": state.get("severity") or "unknown",
        "cost_range": state.get("cost_range") or (0, 0),
        "compensable_amount": state.get("compensable_amount") or (0, 0),
        "deductible_applied": state.get("deductible_applied") or 0,
        "ceiling_applied": state.get("ceiling_applied") or 0,
        "summary": state.get("summary") or "",
        "claim": state["verdict"]["claim"],
    }}


def build_graph(vlm_inference=None, inference=None, attachments_dir: Path | None = None):
    """Build the expertise workflow graph."""
    if attachments_dir is None:
        attachments_dir = Path("data/attachments")

    workflow = StateGraph(ExpertiseState)

    workflow.add_node("assess_severity", make_assess_severity_node(vlm_inference, attachments_dir))
    workflow.add_node("estimate_costs", estimate_costs_node)
    workflow.add_node("generate_report", make_generate_report_node(inference))
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("assess_severity")
    workflow.add_edge("assess_severity", "estimate_costs")
    workflow.add_edge("estimate_costs", "generate_report")
    workflow.add_edge("generate_report", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
