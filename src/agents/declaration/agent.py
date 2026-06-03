import re
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.declaration.prompts import (
    EXTRACT_SYSTEM_PROMPT,
    FOLLOW_UP_SYSTEM_PROMPT,
    follow_up_prompt,
)
from src.agents.declaration.state import ClaimState
from src.agents.declaration.tools import REQUIRED_FIELDS, merge_claim_data, validate_date


def check_completeness(state: ClaimState) -> dict:
    """Determine which required fields are missing and whether the claim is complete."""
    claim = state.get("claim_data", {})
    missing = [f for f in REQUIRED_FIELDS if claim.get(f) is None]
    return {"missing_fields": missing, "is_complete": len(missing) == 0}


def finalize(state: ClaimState) -> dict:
    """Extract photo filenames and assemble the final claim with conversation history."""
    photo_filenames = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            found = re.findall(
                r"\[(?:Pièce jointe|Attachment)\s*:\s*([^\]]+)\]", msg.content
            )
            photo_filenames.extend(found)

    final_claim = {
        **state["claim_data"],
        "photo_filenames": photo_filenames,
        "conversation_history": [
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content,
            }
            for m in state["messages"]
        ],
        "turns_count": sum(1 for m in state["messages"] if isinstance(m, HumanMessage)),
    }
    return {"final_claim": final_claim}


def make_extract_fields_node(inference):
    """Factory function that creates a node to extract claim fields from messages."""
    def extract_fields(state: ClaimState) -> dict:
        """Extract claim fields from the conversation history using the inference model."""
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if not last_human:
            return {}

        history_messages = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                history_messages.append({"role": "user", "content": msg.content})
            else:
                history_messages.append({"role": "assistant", "content": msg.content})

        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT.format(today=date.today().isoformat())},
            *history_messages,
        ]
        new_fields = inference.extract(messages)
        new_fields["date"] = validate_date(new_fields.get("date"))
        merged = merge_claim_data(state.get("claim_data", {}), new_fields)
        return {"claim_data": merged}

    return extract_fields


def make_generate_follow_up_node(inference):
    """Factory function that creates a node to generate follow-up questions."""
    def generate_follow_up(state: ClaimState) -> dict:
        """Generate a follow-up message requesting missing claim information."""
        prompt = follow_up_prompt(state["claim_data"], state["missing_fields"])
        messages = [
            {"role": "system", "content": FOLLOW_UP_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = inference.generate(messages)
        return {"messages": [AIMessage(content=response)]}

    return generate_follow_up


def build_graph(inference):
    """Build the claim processing graph with extraction, completeness check, and finalization nodes."""
    workflow = StateGraph(ClaimState)

    workflow.add_node("extract_fields", make_extract_fields_node(inference))
    workflow.add_node("check_completeness", check_completeness)
    workflow.add_node("generate_follow_up", make_generate_follow_up_node(inference))
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("extract_fields")
    workflow.add_edge("extract_fields", "check_completeness")
    workflow.add_conditional_edges(
        "check_completeness",
        lambda s: "finalize" if s["is_complete"] else "generate_follow_up",
    )
    workflow.add_edge("generate_follow_up", END)
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=MemorySaver())
