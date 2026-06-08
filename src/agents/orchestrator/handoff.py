from langgraph.types import interrupt


def human_gate(role: str, question: str, key: str, state: dict):
    """Return a pre-seeded interlocutor answer if present, else interrupt for live input.

    Pre-seeded mode (``state["human_responses"][key]`` set): returns it, no pause —
    scriptable and reproducible for tests/eval.
    Interactive mode (key absent): ``interrupt()`` suspends the graph; the driver in
    ``main.py`` collects the answer and resumes via ``Command(resume=...)``.
    """
    preseeded = (state.get("human_responses") or {}).get(key)
    if preseeded is not None:
        return preseeded
    return interrupt({"role": role, "question": question, "key": key})
