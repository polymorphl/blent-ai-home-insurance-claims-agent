from datetime import date
from src.agents.validation.state import ValidationState
from src.agents.validation.rules import COVERAGE_RULES, business_days_since


def test_validation_state_has_required_keys():
    state: ValidationState = {
        "claim": {"date": "2025-09-10", "incident_type": "fire",
                  "description": "Incendie.", "has_photos": True},
        "conformity_errors": [],
        "coverage_errors": [],
        "verdict": None,
        "today_override": None,
    }
    assert state["claim"]["incident_type"] == "fire"
    assert state["verdict"] is None


def test_coverage_rules_has_three_incident_types():
    assert set(COVERAGE_RULES.keys()) == {"water_damage", "fire", "theft"}


def test_coverage_rules_water_damage_values():
    rule = COVERAGE_RULES["water_damage"]
    assert rule["ceiling"] == 25000
    assert rule["deductible"] == 150
    assert rule["deadline_days"] == 5


def test_coverage_rules_fire_values():
    rule = COVERAGE_RULES["fire"]
    assert rule["ceiling"] == 100000
    assert rule["deductible"] == 300
    assert rule["deadline_days"] == 5


def test_coverage_rules_theft_values():
    rule = COVERAGE_RULES["theft"]
    assert rule["ceiling"] == 20000
    assert rule["deductible"] == 200
    assert rule["deadline_days"] == 2


def test_business_days_same_day():
    today = date(2026, 6, 4)
    assert business_days_since("2026-06-04", today=today) == 0


def test_business_days_one_weekday():
    # 2026-06-03 (Wednesday) → 2026-06-04 (Thursday) = 1 business day
    assert business_days_since("2026-06-03", today=date(2026, 6, 4)) == 1


def test_business_days_skips_weekend():
    # 2026-06-05 (Friday) → 2026-06-08 (Monday) = 1 business day (Fri counted, weekend skipped)
    assert business_days_since("2026-06-05", today=date(2026, 6, 8)) == 1


def test_business_days_two_weeks():
    # 2026-06-01 (Monday) → 2026-06-15 (Monday) = 10 business days
    assert business_days_since("2026-06-01", today=date(2026, 6, 15)) == 10


def test_business_days_defaults_to_today():
    # Should not raise and should return >= 0
    result = business_days_since("2026-06-01")
    assert result >= 0
