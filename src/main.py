from langgraph.types import Command

from src.agents.declaration.agent import build_graph as build_declaration_graph
from src.agents.validation.agent import build_graph as build_validation_graph
from src.agents.expertise.agent import build_graph as build_expertise_graph
from src.agents.orchestrator.agent import build_orchestrator_graph
from src.examples import EXAMPLES
from src.inference import UnifiedInference


def resolve_interrupt(req: dict):
    """Prompt the operator on the console for an interlocutor's manual answer."""
    print(f"\n[INTERVENTION REQUISE — {req['role']}]")
    print(f"  {req['question']}")
    return input("  > ")


def run_pipeline(app, example: dict) -> dict:
    """Run one claim end-to-end, resolving any human-in-the-loop interrupts."""
    print(f"\n{'=' * 60}")
    print(f"Example: {example['id']}")
    print(f"{'=' * 60}")

    config = {"configurable": {"thread_id": example["id"]}}
    state = {
        "example": example,
        "today_override": example.get("today_override"),
        "human_responses": example.get("human_responses", {}),
        "final_claim": None, "verdict": None, "report": None,
        "quotes": {}, "decision": None,
    }

    result = app.invoke(state, config)
    while "__interrupt__" in result:
        req = result["__interrupt__"][0].value
        answer = resolve_interrupt(req)
        result = app.invoke(Command(resume=answer), config)

    _print_result(result)
    return result


def _print_result(result: dict) -> None:
    verdict = result.get("verdict") or {}
    report = result.get("report")
    decision = result.get("decision") or {}

    print(f"\n  Verdict:  {verdict.get('status', 'n/a').upper()} — {verdict.get('reason', '')}")
    if report:
        print(f"  Severity: {report['severity']}")
        print(f"  Cost:     {report['cost_range'][0]}€ – {report['cost_range'][1]}€")
        print(f"  Comp.:    {report['compensable_amount'][0]}€ – {report['compensable_amount'][1]}€")
    print(f"  Decision: approved={decision.get('approved')}  amount={decision.get('final_amount')}")
    if decision.get("quotes"):
        print(f"  Quotes:   {decision['quotes']}")
    print(f"  Note:     {decision.get('advisor_note', '')}")


def main():
    """Load models and run the full orchestrated pipeline on all examples."""
    model = UnifiedInference()
    declaration_app = build_declaration_graph(model)
    validation_app = build_validation_graph(vlm_inference=model)
    expertise_app = build_expertise_graph(vlm_inference=model, inference=model)
    app = build_orchestrator_graph(declaration_app, validation_app, expertise_app)

    for example in EXAMPLES:
        run_pipeline(app, example)


if __name__ == "__main__":
    main()
