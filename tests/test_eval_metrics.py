from src.eval.metrics import (
    declaration_completeness,
    validation_factual,
    orchestration_provider,
)


def test_declaration_completeness_full():
    extracted = {"date": "2026-06-01", "incident_type": "water_damage",
                 "description": "x", "has_photos": True}
    m = declaration_completeness(extracted)
    assert m["score"] == 1.0
    assert m["fields_ok"] == "4/4"
    assert all(m["per_field"].values())


def test_declaration_completeness_partial():
    extracted = {"date": "2026-06-01", "incident_type": "water_damage"}
    m = declaration_completeness(extracted)
    assert m["score"] == 0.5
    assert m["fields_ok"] == "2/4"
    assert m["per_field"]["description"] is False


def test_declaration_completeness_empty():
    m = declaration_completeness({})
    assert m["score"] == 0.0
    assert m["fields_ok"] == "0/4"


def test_validation_factual_approved_match():
    verdict = {"status": "approved", "coverage": {"ceiling": 25000, "deductible": 150}}
    expected = {"status": "approved", "ceiling": 25000, "deductible": 150}
    m = validation_factual(verdict, expected)
    assert m["status_correct"] is True
    assert m["coverage_correct"] is True


def test_validation_factual_wrong_coverage():
    verdict = {"status": "approved", "coverage": {"ceiling": 999, "deductible": 150}}
    expected = {"status": "approved", "ceiling": 25000, "deductible": 150}
    m = validation_factual(verdict, expected)
    assert m["status_correct"] is True
    assert m["coverage_correct"] is False


def test_validation_factual_rejected_match():
    verdict = {"status": "rejected", "coverage": None}
    expected = {"status": "rejected"}
    m = validation_factual(verdict, expected)
    assert m["status_correct"] is True
    assert m["coverage_correct"] is True


def test_validation_factual_wrong_status():
    verdict = {"status": "approved", "coverage": {"ceiling": 25000, "deductible": 150}}
    expected = {"status": "rejected"}
    m = validation_factual(verdict, expected)
    assert m["status_correct"] is False


def test_orchestration_provider():
    assert orchestration_provider("plombier", "plombier")["correct"] is True
    bad = orchestration_provider("vitrier", "serrurier")
    assert bad["correct"] is False
    assert bad["provider"] == "vitrier"


from src.eval.harness import load_golden_dataset, run_declaration_case, run_validation_case


class _SeqDecl:
    """Fake declaration app returning a sequence of outputs over successive .invoke calls."""
    def __init__(self, outputs):
        self.outputs = outputs
        self.n = 0

    def invoke(self, state, config=None):
        out = self.outputs[min(self.n, len(self.outputs) - 1)]
        self.n += 1
        return out


class _FakeVal:
    def __init__(self):
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append(state)
        return {"verdict": {"status": "approved",
                            "coverage": {"ceiling": 25000, "deductible": 150},
                            "claim": state["claim"]}}


def test_load_golden_dataset_returns_nine():
    assert len(load_golden_dataset()) == 9


def test_run_declaration_case_drives_all_turns_until_complete():
    claim = {"date": "2026-06-01", "incident_type": "fire",
             "description": "f", "has_photos": True}
    decl = _SeqDecl([
        {"is_complete": False, "claim_data": {"incident_type": "fire"}},
        {"is_complete": True, "final_claim": claim, "claim_data": claim},
    ])
    case = {"id": "c", "turns": [{"role": "user", "content": "a"},
                                 {"role": "user", "content": "b"}]}
    extracted = run_declaration_case(decl, case)
    assert decl.n == 2
    assert extracted["incident_type"] == "fire"


def test_run_declaration_case_returns_claim_data_when_incomplete():
    decl = _SeqDecl([{"is_complete": False, "claim_data": {"incident_type": "theft"}}])
    case = {"id": "c", "turns": [{"role": "user", "content": "a"}]}
    extracted = run_declaration_case(decl, case)
    assert extracted == {"incident_type": "theft"}


def test_run_validation_case_uses_gold_claim_and_today_override():
    case = {"id": "c",
            "expected": {"declaration": {"date": "2026-06-01", "incident_type": "water_damage",
                                         "description": "x", "has_photos": True}},
            "today_override": "2026-06-04"}
    val = _FakeVal()
    verdict = run_validation_case(val, case)
    assert val.calls[0]["today_override"] == "2026-06-04"
    assert val.calls[0]["claim"]["incident_type"] == "water_damage"
    assert verdict["status"] == "approved"


from src.eval.report import build_row, aggregates, write_csv, CSV_COLUMNS


def _sample_inputs():
    case = {"id": "x", "family": "water_damage",
            "expected": {"validation": {"status": "approved"},
                         "orchestration": {"provider": "plombier"}}}
    decl_m = {"score": 1.0, "fields_ok": "4/4", "per_field": {}}
    val_m = {"status": "approved", "status_correct": True, "coverage_correct": True}
    orch_m = {"provider": "plombier", "correct": True}
    return case, decl_m, val_m, orch_m


def test_build_row_shape():
    row = build_row(*_sample_inputs())
    assert set(row.keys()) == set(CSV_COLUMNS)
    assert row["id"] == "x"
    assert row["decl_completeness"] == 1.0
    assert row["val_status_expected"] == "approved"
    assert row["orch_provider_correct"] is True


def test_aggregates():
    row = build_row(*_sample_inputs())
    agg = aggregates([row])
    assert agg["decl_completeness_mean"] == 1.0
    assert agg["val_accuracy"] == 1.0
    assert agg["orch_precision"] == 1.0


def test_write_csv(tmp_path):
    row = build_row(*_sample_inputs())
    p = tmp_path / "out.csv"
    write_csv([row], p)
    text = p.read_text()
    assert ",".join(CSV_COLUMNS) in text
    assert "water_damage" in text


import importlib


def test_model_free_scripts_run(capsys):
    # validation + orchestration scripts need no model — run their main() over the dataset.
    val = importlib.import_module("scripts.evaluate_validation")
    orch = importlib.import_module("scripts.evaluate_orchestration")
    val.main()
    orch.main()
    out = capsys.readouterr().out
    assert "Validation accuracy" in out
    assert "Provider precision" in out


def test_model_dependent_scripts_import():
    # declaration + consolidator import cleanly without instantiating a model.
    importlib.import_module("scripts.evaluate_declaration")
    importlib.import_module("scripts.evaluate_agents")
