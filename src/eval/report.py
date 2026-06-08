import csv
from pathlib import Path

CSV_COLUMNS = [
    "id", "family",
    "decl_completeness", "decl_fields_ok",
    "val_status", "val_status_expected", "val_status_correct", "val_coverage_correct",
    "orch_provider", "orch_provider_expected", "orch_provider_correct",
]


def build_row(case: dict, decl_m: dict, val_m: dict, orch_m: dict) -> dict:
    """Assemble one CSV row from a case and its three per-agent metric dicts."""
    exp = case["expected"]
    return {
        "id": case["id"],
        "family": case["family"],
        "decl_completeness": round(decl_m["score"], 2),
        "decl_fields_ok": decl_m["fields_ok"],
        "val_status": val_m["status"],
        "val_status_expected": exp["validation"]["status"],
        "val_status_correct": val_m["status_correct"],
        "val_coverage_correct": val_m["coverage_correct"],
        "orch_provider": orch_m["provider"],
        "orch_provider_expected": exp["orchestration"]["provider"],
        "orch_provider_correct": orch_m["correct"],
    }


def aggregates(rows: list[dict]) -> dict:
    """Mean completeness, validation accuracy, orchestration precision."""
    n = len(rows) or 1
    return {
        "decl_completeness_mean": round(sum(r["decl_completeness"] for r in rows) / n, 3),
        "val_accuracy": round(sum(1 for r in rows if r["val_status_correct"]) / n, 3),
        "orch_precision": round(sum(1 for r in rows if r["orch_provider_correct"]) / n, 3),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    """Write rows to a CSV file with the canonical column order."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
