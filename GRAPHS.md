# Agent Graphs

## Declaration Agent

```mermaid
flowchart TD
    START([Start]) --> extract_fields

    extract_fields["extract_fields\n(LLM tool call)"]
    check_completeness{"check_completeness\nall fields present?"}
    generate_follow_up["generate_follow_up\n(LLM)"]
    finalize_decl["finalize\nassemble final_claim"]

    extract_fields --> check_completeness
    check_completeness -->|complete| finalize_decl
    check_completeness -->|missing fields| generate_follow_up
    generate_follow_up --> END_TURN([End — await next turn])
    finalize_decl --> END_DECL([End])
```

**Input:** policyholder message  
**Output:** `final_claim` — `{date, incident_type, description, has_photos, photo_filenames, conversation_history}`  
**Model:** Qwen2.5-7B-Instruct (tool calling + text generation)

---

## Validation Agent

```mermaid
flowchart TD
    START([Start]) --> check_conformity

    check_conformity["check_conformity\nfields present? photos required?"]
    check_coverage["check_coverage\ntype covered? deadline respected?"]
    finalize_val["finalize\nbuild verdict"]

    check_conformity -->|errors| finalize_val
    check_conformity -->|ok| check_coverage
    check_coverage --> finalize_val
    finalize_val --> END([End])
```

**Input:** `final_claim` from Declaration Agent + `today_override` (optional)  
**Output:** `verdict` — `{status, reason, coverage, claim}`  
**Model:** none — pure rule-based
