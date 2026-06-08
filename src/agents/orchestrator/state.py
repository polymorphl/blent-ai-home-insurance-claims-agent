from typing import TypedDict


class OrchestratorState(TypedDict, total=False):
    """State for the end-to-end claims pipeline orchestrator."""
    # inputs
    example: dict                  # {id, turns, ...} drives the declaration sub-run
    today_override: str | None     # forwarded to validation
    human_responses: dict          # pre-seeded interlocutor answers, keyed by gate key
    # intermediate
    final_claim: dict | None
    verdict: dict | None
    report: dict | None
    quotes: dict                   # accumulated provider inputs (plombier/expert/serrurier)
    # output
    decision: dict | None
