from langchain_core.messages import AIMessage, HumanMessage

from src.agents.declaration.agent import build_graph
from src.agents.declaration.inference import HFInference
from src.examples import EXAMPLES


def run_example(app, example: dict) -> dict | None:
    """Execute a claim processing example through the agent, printing progress and results."""
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
            print(f"\n✓ Complete after {i + 1} turn(s)")
            print(f"  Final claim: {result['final_claim']}")
            return result["final_claim"]

        last_ai = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai:
            print(f"\n[Agent] {last_ai.content}")

    print("\n✗ Incomplete after all simulated turns")
    return None


def main():
    """Load the model and run all example claim scenarios."""
    print("⌛ Loading model...")
    inference = HFInference()
    app = build_graph(inference)

    for example in EXAMPLES:
        run_example(app, example)


if __name__ == "__main__":
    main()
