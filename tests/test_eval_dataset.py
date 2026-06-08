import json
from pathlib import Path

from src.agents.validation.agent import build_graph as build_validation_graph
from src.agents.orchestrator.agent import select_provider
from src.eval.harness import load_golden_dataset, run_validation_case
from src.eval.metrics import validation_factual, orchestration_provider

DATASET = Path(__file__).resolve().parents[1] / "data" / "golden_dataset_full.json"
PROVIDER = {"water_damage": "plombier", "fire": "expert", "theft": "serrurier"}


def test_dataset_has_nine_cases_three_per_family():
    data = json.loads(DATASET.read_text())
    assert len(data) == 9
    families = [c["family"] for c in data]
    for fam in ("water_damage", "fire", "theft"):
        assert families.count(fam) == 3


def test_dataset_case_shape():
    data = json.loads(DATASET.read_text())
    ids = set()
    for c in data:
        assert {"id", "family", "turns", "today_override", "expected"} <= set(c)
        ids.add(c["id"])
        exp = c["expected"]
        assert {"declaration", "validation", "orchestration"} <= set(exp)
        assert exp["declaration"]["incident_type"] == c["family"]
        assert exp["orchestration"]["provider"] == PROVIDER[c["family"]]
        assert exp["validation"]["status"] in ("approved", "rejected")
        assert c["turns"] and all("content" in t for t in c["turns"])
    assert len(ids) == 9  # unique ids


def test_validation_labels_match_rules():
    app = build_validation_graph()
    for case in load_golden_dataset():
        verdict = run_validation_case(app, case)
        m = validation_factual(verdict, case["expected"]["validation"])
        assert m["status_correct"], f"{case['id']}: got {m['status']}"
        assert m["coverage_correct"], f"{case['id']}: coverage mismatch"


def test_orchestration_labels_match_mapping():
    for case in load_golden_dataset():
        chosen = select_provider(case["expected"]["declaration"]["incident_type"])
        m = orchestration_provider(chosen, case["expected"]["orchestration"]["provider"])
        assert m["correct"], case["id"]
