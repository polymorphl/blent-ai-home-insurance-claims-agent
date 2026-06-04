from langchain_core.messages import AIMessage, HumanMessage

from src.agents.declaration.agent import build_graph as build_declaration_graph
from src.agents.declaration.inference import HFInference
from src.agents.validation.agent import build_graph as build_validation_graph
from src.examples import EXAMPLES


def run_declaration(app, example: dict) -> dict | None:
    """Run the Declaration Agent on a simulated example, returning final_claim or None."""
    print(f"\n{'=' * 60}")
    print(f"Example: {example['id']}")
    print(f"{'=' * 60}")

    config = {"configurable": {"thread_id": example["id"]}}
    first_turn = True

    for i, content in enumerate(example["turns"]):
        print(f"\n[Turn {i + 1}] User: {content[:100]}...")

        if first_turn:
            state = {
                "messages": [HumanMessage(content=content)],
                "claim_data": {},
                "missing_fields": [],
                "is_complete": False,
                "final_claim": None,
            }
            first_turn = False
        else:
            state = {"messages": [HumanMessage(content=content)]}

        result = app.invoke(state, config)

        if result["is_complete"]:
            print(f"\n✓ Declaration complete after {i + 1} turn(s)")
            return result["final_claim"]

        last_ai = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai:
            print(f"\n[Agent] {last_ai.content}")

    print("\n✗ Declaration incomplete after all simulated turns")
    return None


def run_validation(app, final_claim: dict) -> dict:
    """Run the Validation Agent on a completed claim and print the verdict."""
    print("\n--- Validation ---")
    result = app.invoke({
        "claim": final_claim,
        "conformity_errors": [],
        "coverage_errors": [],
        "verdict": None,
        "today_override": None,
    })
    verdict = result["verdict"]
    status = verdict["status"].upper()
    print(f"  Status:   {status}")
    print(f"  Reason:   {verdict['reason']}")
    if verdict["coverage"]:
        print(f"  Coverage: ceiling={verdict['coverage']['ceiling']}€  deductible={verdict['coverage']['deductible']}€")
    return verdict


def main():
    """Load the model and run the full pipeline (Declaration → Validation) on all examples."""
    inference = HFInference()
    declaration_app = build_declaration_graph(inference)
    validation_app = build_validation_graph()

    for example in EXAMPLES:
        final_claim = run_declaration(declaration_app, example)
        if final_claim:
            run_validation(validation_app, final_claim)


if __name__ == "__main__":
    main()
