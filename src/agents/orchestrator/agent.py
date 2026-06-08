from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.orchestrator.handoff import human_gate
from src.agents.orchestrator.state import OrchestratorState

# incident_type → service provider (per data/...Processus.md):
# water → plumber, fire → expert, theft → locksmith
PROVIDER_BY_TYPE = {"water_damage": "plombier", "fire": "expert", "theft": "serrurier"}


def select_provider(incident_type: str | None) -> str | None:
    """Return the service provider for an incident type, or None if unmapped."""
    return PROVIDER_BY_TYPE.get(incident_type)


def make_declaration_node(declaration_app):
    """Drive the multi-turn declaration subgraph over example['turns']."""
    def declaration_node(state: OrchestratorState) -> dict:
        example = state["example"]
        sub_config = {"configurable": {"thread_id": f"decl_{example['id']}"}}
        final_claim = None
        first = True
        for content in example["turns"]:
            if first:
                sub_state = {
                    "messages": [HumanMessage(content=content)],
                    "claim_data": {}, "missing_fields": [],
                    "is_complete": False, "final_claim": None,
                }
                first = False
            else:
                sub_state = {"messages": [HumanMessage(content=content)]}
            result = declaration_app.invoke(sub_state, sub_config)
            if result.get("is_complete"):
                final_claim = result.get("final_claim")
                break
        return {"final_claim": final_claim}

    return declaration_node


def make_validation_node(validation_app):
    """Invoke the validation subgraph on the collected claim."""
    def validation_node(state: OrchestratorState) -> dict:
        final_claim = state.get("final_claim")
        if final_claim is None:
            return {"verdict": {
                "status": "rejected",
                "reason": "Déclaration incomplète après tous les échanges.",
                "coverage": None, "claim": {},
            }}
        result = validation_app.invoke({
            "claim": final_claim,
            "conformity_errors": [], "coverage_errors": [], "photo_errors": [],
            "verdict": None, "today_override": state.get("today_override"),
        })
        return {"verdict": result["verdict"]}

    return validation_node


def route_after_validation(state: OrchestratorState) -> str:
    return "expertise" if state["verdict"]["status"] == "approved" else "finalize_rejected"


def finalize_rejected(state: OrchestratorState) -> dict:
    verdict = state["verdict"]
    return {"decision": {
        "approved": False, "final_amount": (0, 0),
        "advisor_note": verdict["reason"], "quotes": {},
    }}


def make_expertise_node(expertise_app):
    """Invoke the expertise subgraph on the approved verdict."""
    def expertise_node(state: OrchestratorState) -> dict:
        result = expertise_app.invoke({
            "verdict": state["verdict"], "severity": None, "cost_range": None,
            "compensable_amount": None, "deductible_applied": None,
            "ceiling_applied": None, "summary": None, "report": None,
        })
        return {"report": result["report"]}

    return expertise_node


def _claim_context(report: dict) -> str:
    """Human-readable claim summary shown to an interlocutor before they answer a gate."""
    claim = report.get("claim", {})
    cost = report.get("cost_range") or (0, 0)
    return (
        "Contexte du sinistre :\n"
        f"  - Type        : {claim.get('incident_type')}\n"
        f"  - Date        : {claim.get('date')}\n"
        f"  - Description : {claim.get('description')}\n"
        f"  - Sévérité    : {report.get('severity')}\n"
        f"  - Coût estimé : {cost[0]}–{cost[1]}€"
    )


def quote_gate(state: OrchestratorState) -> dict:
    """Collect input from the relevant provider (plombier/expert/serrurier), if any."""
    report = state["report"]
    role = select_provider(report["claim"].get("incident_type"))
    if role is None:
        return {}
    question = f"{_claim_context(report)}\nDevis {role} (montant ou note) pour ce sinistre ?"
    answer = human_gate(role, question, role, state)
    return {"quotes": {**state.get("quotes", {}), role: answer}}


def expert_gate(state: OrchestratorState) -> dict:
    """Request a field expert opinion when severity is high."""
    report = state["report"]
    if report.get("severity") != "high":
        return {}
    question = f"{_claim_context(report)}\nAvis d'expert terrain (sévérité élevée) ?"
    answer = human_gate("expert", question, "expert", state)
    return {"quotes": {**state.get("quotes", {}), "expert": answer}}


def _build_decision(answer, report, quotes: dict) -> dict:
    """Normalize the conseiller answer (dict or plain string) into a decision dict."""
    if isinstance(answer, dict):
        return {
            "approved": answer.get("approved", True),
            "final_amount": tuple(answer.get("final_amount", report.get("compensable_amount") or (0, 0))),
            "advisor_note": answer.get("advisor_note", ""),
            "quotes": quotes,
        }
    return {
        "approved": True,
        "final_amount": report.get("compensable_amount") or (0, 0),
        "advisor_note": str(answer),
        "quotes": quotes,
    }


def advisor_gate(state: OrchestratorState) -> dict:
    """Conseiller makes the final compensation decision."""
    report = state["report"]
    comp = report.get("compensable_amount") or (0, 0)
    quotes = state.get("quotes", {})
    question = (
        f"{_claim_context(report)}\n"
        f"Décision finale (conseiller). Montant indemnisable estimé : {comp[0]}–{comp[1]}€. "
        f"Devis reçus : {quotes}. "
        f"Répondre par un objet {{approved, final_amount, advisor_note}} ou une note libre."
    )
    answer = human_gate("conseiller", question, "conseiller", state)
    return {"decision": _build_decision(answer, report, quotes)}


def build_orchestrator_graph(declaration_app, validation_app, expertise_app):
    """Compose the three compiled subgraphs into the end-to-end pipeline."""
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("declaration", make_declaration_node(declaration_app))
    workflow.add_node("validation", make_validation_node(validation_app))
    workflow.add_node("finalize_rejected", finalize_rejected)
    workflow.add_node("expertise", make_expertise_node(expertise_app))
    workflow.add_node("quote_gate", quote_gate)
    workflow.add_node("expert_gate", expert_gate)
    workflow.add_node("advisor_gate", advisor_gate)

    workflow.set_entry_point("declaration")
    workflow.add_edge("declaration", "validation")
    workflow.add_conditional_edges(
        "validation", route_after_validation,
        {"expertise": "expertise", "finalize_rejected": "finalize_rejected"},
    )
    workflow.add_edge("finalize_rejected", END)
    workflow.add_edge("expertise", "quote_gate")
    workflow.add_edge("quote_gate", "expert_gate")
    workflow.add_edge("expert_gate", "advisor_gate")
    workflow.add_edge("advisor_gate", END)

    return workflow.compile(checkpointer=MemorySaver())
