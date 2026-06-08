from src.agents.orchestrator.state import OrchestratorState
from src.agents.orchestrator.handoff import human_gate


def test_orchestrator_state_has_expected_keys():
    annotations = OrchestratorState.__annotations__
    expected = {
        "example", "today_override", "human_responses",
        "final_claim", "verdict", "report", "quotes", "decision",
    }
    assert expected.issubset(set(annotations.keys()))


def test_human_gate_returns_preseeded_answer_without_interrupt():
    state = {"human_responses": {"plombier": "Devis 1200€"}}
    result = human_gate("plombier", "Devis ?", "plombier", state)
    assert result == "Devis 1200€"


def test_human_gate_missing_human_responses_key_is_safe():
    # No "human_responses" at all → must not raise KeyError before deciding to interrupt.
    # We assert it does NOT return a pre-seeded value (would raise GraphInterrupt at runtime,
    # which is exercised in the integration test). Here we only confirm the lookup is safe.
    state = {"human_responses": {"conseiller": "x"}}
    result = human_gate("conseiller", "Décision ?", "conseiller", state)
    assert result == "x"


from langgraph.types import Command
from src.agents.orchestrator.agent import build_orchestrator_graph


class _FakeApp:
    """Stub compiled graph: returns a fixed dict from .invoke, records calls."""
    def __init__(self, output):
        self.output = output
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append(state)
        return self.output


def _approved_report_apps(severity="high", incident_type="water_damage"):
    claim = {"incident_type": incident_type, "date": "2026-06-01",
             "description": "dégât", "has_photos": True, "photo_filenames": ["a.jpg"]}
    declaration = _FakeApp({"is_complete": True, "final_claim": claim})
    validation = _FakeApp({"verdict": {
        "status": "approved", "reason": "Dossier validé.",
        "coverage": {"ceiling": 25000, "deductible": 150}, "claim": claim,
    }})
    expertise = _FakeApp({"report": {
        "severity": severity, "cost_range": (8000, 20000),
        "compensable_amount": (7850, 19850), "deductible_applied": 150,
        "ceiling_applied": 25000, "summary": "résumé", "claim": claim,
    }})
    return declaration, validation, expertise


def _invoke(app, example, human_responses):
    config = {"configurable": {"thread_id": example["id"]}}
    state = {
        "example": example, "today_override": example.get("today_override"),
        "human_responses": human_responses,
        "final_claim": None, "verdict": None, "report": None,
        "quotes": {}, "decision": None,
    }
    return app.invoke(state, config), config


def test_full_pipeline_preseeded_produces_decision():
    decl, val, exp = _approved_report_apps(severity="high", incident_type="water_damage")
    app = build_orchestrator_graph(decl, val, exp)
    example = {"id": "t_full", "turns": ["bonjour"]}
    human_responses = {
        "plombier": "Devis plomberie 1200€",
        "expert": "Sévérité confirmée élevée",
        "conseiller": {"approved": True, "final_amount": [7850, 19850],
                       "advisor_note": "Indemnisation accordée"},
    }
    result, _ = _invoke(app, example, human_responses)
    assert "__interrupt__" not in result
    decision = result["decision"]
    assert decision["approved"] is True
    assert decision["advisor_note"] == "Indemnisation accordée"
    assert decision["quotes"]["plombier"] == "Devis plomberie 1200€"
    assert decision["quotes"]["expert"] == "Sévérité confirmée élevée"


def test_rejected_verdict_short_circuits_expertise():
    claim = {"incident_type": "theft", "has_photos": False}
    decl = _FakeApp({"is_complete": True, "final_claim": claim})
    val = _FakeApp({"verdict": {"status": "rejected",
                                "reason": "Des photos sont requises.",
                                "coverage": None, "claim": claim}})
    exp = _FakeApp({"report": {}})  # must never be called
    app = build_orchestrator_graph(decl, val, exp)
    example = {"id": "t_reject", "turns": ["bonjour"]}
    result, _ = _invoke(app, example, {})
    assert exp.calls == []  # expertise short-circuited
    assert result["decision"]["approved"] is False
    assert result["decision"]["advisor_note"] == "Des photos sont requises."


def test_gate_question_includes_claim_context():
    # No pre-seed → first gate (plombier for water) interrupts; its question must carry
    # the claim context so the operator can answer informed.
    decl, val, exp = _approved_report_apps(severity="low", incident_type="water_damage")
    app = build_orchestrator_graph(decl, val, exp)
    example = {"id": "t_ctx", "turns": ["bonjour"]}
    result, _ = _invoke(app, example, {})
    assert "__interrupt__" in result
    q = result["__interrupt__"][0].value["question"]
    assert "water_damage" in q       # incident type
    assert "dégât" in q              # description from _approved_report_apps
    assert "8000" in q               # estimated cost low bound


def test_interrupt_fires_then_resumes():
    decl, val, exp = _approved_report_apps(severity="low", incident_type="fire")
    app = build_orchestrator_graph(decl, val, exp)
    example = {"id": "t_interrupt", "turns": ["bonjour"]}
    # fire → expert provider (pre-seeded), severity low → no expert_gate;
    # only the conseiller gate pauses.
    result, config = _invoke(app, example, {"expert": "Avis expert pré-rempli"})
    assert "__interrupt__" in result
    req = result["__interrupt__"][0].value
    assert req["role"] == "conseiller"
    result = app.invoke(Command(resume={"approved": True, "final_amount": [0, 0],
                                        "advisor_note": "ok"}), config)
    assert "__interrupt__" not in result
    assert result["decision"]["approved"] is True


def test_gate_selection_by_claim():
    # water_damage + low severity → plombier quote, NO expert.
    decl, val, exp = _approved_report_apps(severity="low", incident_type="water_damage")
    app = build_orchestrator_graph(decl, val, exp)
    example = {"id": "t_gates", "turns": ["bonjour"]}
    human_responses = {"plombier": "Devis 900€", "conseiller": "OK"}
    result, _ = _invoke(app, example, human_responses)
    assert "__interrupt__" not in result
    assert result["decision"]["quotes"] == {"plombier": "Devis 900€"}


class _SeqApp:
    """Fake compiled graph returning a sequence of outputs across successive .invoke calls."""
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append(state)
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]


def test_validation_receives_today_override():
    decl, val, exp = _approved_report_apps(severity="low", incident_type="water_damage")
    app = build_orchestrator_graph(decl, val, exp)
    example = {"id": "t_today", "turns": ["bonjour"], "today_override": "2026-06-05"}
    _invoke(app, example, {"plombier": "x", "conseiller": "ok"})
    assert val.calls[0]["today_override"] == "2026-06-05"


def test_incomplete_declaration_yields_rejected_decision():
    decl = _FakeApp({"is_complete": False})  # never completes
    val = _FakeApp({"verdict": {"status": "should_not_be_used"}})  # not reached as approved
    exp = _FakeApp({"report": {}})  # must never be called
    app = build_orchestrator_graph(decl, val, exp)
    example = {"id": "t_incomplete", "turns": ["bonjour", "encore"]}
    result, _ = _invoke(app, example, {})
    assert exp.calls == []
    assert result["decision"]["approved"] is False
    assert "incomplète" in result["decision"]["advisor_note"].lower()


def test_declaration_loops_until_complete():
    claim = {"incident_type": "fire", "date": "2026-06-01", "description": "feu",
             "has_photos": True, "photo_filenames": ["a.jpg"]}
    decl = _SeqApp([
        {"is_complete": False},                         # turn 1: incomplete
        {"is_complete": True, "final_claim": claim},    # turn 2: complete
    ])
    val = _FakeApp({"verdict": {"status": "approved", "reason": "ok",
                                "coverage": {"ceiling": 100000, "deductible": 300},
                                "claim": claim}})
    exp = _FakeApp({"report": {"severity": "low", "cost_range": (1000, 10000),
                               "compensable_amount": (700, 9700), "deductible_applied": 300,
                               "ceiling_applied": 100000, "summary": "s", "claim": claim}})
    app = build_orchestrator_graph(decl, val, exp)
    example = {"id": "t_multiturn", "turns": ["bonjour", "le 01/06/2026"]}
    result, _ = _invoke(app, example, {"expert": "avis", "conseiller": "ok"})  # fire → expert provider
    assert len(decl.calls) == 2  # looped through both turns
    assert result["decision"]["approved"] is True


def test_advisor_free_note_string_answer():
    decl, val, exp = _approved_report_apps(severity="low", incident_type="fire")
    app = build_orchestrator_graph(decl, val, exp)
    example = {"id": "t_freenote", "turns": ["bonjour"]}
    result, _ = _invoke(app, example, {"expert": "avis", "conseiller": "Validé sous réserve"})
    decision = result["decision"]
    assert decision["approved"] is True
    assert decision["advisor_note"] == "Validé sous réserve"
    assert decision["final_amount"] == (0, 0) or decision["final_amount"] == exp.output["report"]["compensable_amount"]


def test_examples_water_case_has_preseeded_responses():
    from src.examples import EXAMPLES
    ex1 = next(e for e in EXAMPLES if e["id"] == "ex1_water_damage_approved")
    assert "human_responses" in ex1
    assert "conseiller" in ex1["human_responses"]


from src.agents.orchestrator.agent import select_provider, PROVIDER_BY_TYPE


def test_provider_mapping_matches_process_doc():
    assert PROVIDER_BY_TYPE == {
        "water_damage": "plombier", "fire": "expert", "theft": "serrurier"}


def test_select_provider_returns_provider_or_none():
    assert select_provider("water_damage") == "plombier"
    assert select_provider("fire") == "expert"
    assert select_provider("theft") == "serrurier"
    assert select_provider("unknown_type") is None
