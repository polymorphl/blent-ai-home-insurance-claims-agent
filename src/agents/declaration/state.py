from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ClaimData(TypedDict, total=False):
    """Typed dictionary containing extracted insurance claim fields."""
    date: str | None
    incident_type: str | None
    description: str | None
    has_photos: bool | None


class ClaimState(TypedDict):
    """State representation for the claim processing workflow."""
    messages: Annotated[list[BaseMessage], add_messages]
    claim_data: ClaimData
    missing_fields: list[str]
    is_complete: bool
    final_claim: dict | None
