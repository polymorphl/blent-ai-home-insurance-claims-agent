# AI Agent - Home Insurance Claims (AssurHabitat)

Multi-agent AI pipeline to automate home insurance claims processing for AssurHabitat.

---

## Context

AssurHabitat, a non-life insurance company specializing in home coverage, wants to streamline and automate claims handling. Manual processing causes significant delays, processing errors, and high operational load — impacting both regulatory compliance and policyholder satisfaction.

This project implements a system of specialized AI agents to guide policyholders through their declaration and automate key steps in the claims workflow.

### Covered claim types

Out of all possible claim types, this project focuses on three:

- **Water damage** (leaks, infiltration, pipe bursts, overflow…)
- **Fire / explosion**
- **Theft, burglary, vandalism**

---

## Agent Architecture

| Agent | Role |
|---|---|
| **Declaration Agent** | Guides the policyholder through filing (photos, documents, legal deadlines) |
| **Validation Agent** | Automatically verifies contract coverage against the declared claim |
| **Expertise Agent** | Estimates repair costs via image analysis (leaks, cracks, material damage) |

Agents are orchestrated by a **parent LangGraph pipeline** (`src/agents/orchestrator/`)
that chains them with conditional branching and human-in-the-loop gates for external
interlocutors (plombier, expert, serrurier, conseiller). See the Orchestration section below.

See [GRAPHS.md](GRAPHS.md) for Mermaid diagrams of each agent's internal flow.

---

## Orchestration

The full pipeline is a parent LangGraph (`src/agents/orchestrator/`) composing the three
agent subgraphs as nodes:

```
declaration → validation ─┬─ rejected ───────────────► decision (rejected)
                          └─ approved → expertise → quote_gate → expert_gate → advisor_gate → decision
```

**Human-in-the-loop.** When an agent needs an external interlocutor, a gate node either:

- injects a **pre-seeded** answer from the scenario's `human_responses` dict (reproducible,
  used by tests/eval), or
- **`interrupt()`s** the graph for **live console input**, resumed via `Command(resume=...)`.

| Gate | Interlocutor | Triggered when |
|---|---|---|
| `quote_gate` | plombier (water) / expert (fire) / serrurier (theft) | by incident type |
| `expert_gate` | expert | `severity == "high"` |
| `advisor_gate` | conseiller | always — final compensation decision |

The pipeline output is a `decision` dict:

```python
{
  "approved": bool,
  "final_amount": (int, int),  # 2-tuple; (0, 0) when rejected
  "advisor_note": str,
  "quotes": {"plombier"?: ..., "expert"?: ..., "serrurier"?: ...},
}
```

---

## Constraints

- **Data stays on-premise** — no calls to OpenAI, Anthropic, or any external API; all inference runs locally
- **Open-weight models only** — Qwen2.5-7B-Instruct for text (tool calling + generation) and Qwen2.5-VL-7B-Instruct for vision (photo coherence + damage severity), with no domain-specific fine-tuning
- **GPU required** — Declaration Agent (Qwen2.5-7B-Instruct) and Validation Agent (Qwen2.5-VL-7B-Instruct for photo coherence) both require a CUDA/MPS-capable device
- **No UI** — the pipeline runs asynchronously in the background, one graph invocation per claim
- **Structured handoff** — each agent outputs a typed dict consumed directly by the next; no free-text passing between agents

---

## Run

Install `uv` if needed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:
```bash
uv sync
```

Set up environment variables:
```bash
cp .env.example .env
# Add your HuggingFace token to .env
```

Run the orchestrated pipeline on all simulated examples (Declaration → Validation →
Expertise → human gates → decision):
```bash
uv run python -m src.main
```

Scenarios with a `human_responses` dict run unattended; others pause at each gate and
prompt for the interlocutor's answer on the console.

Evaluate agents against the golden dataset:
```bash
# Model-free (run anywhere):
uv run python scripts/evaluate_validation.py       # validation factual accuracy
uv run python scripts/evaluate_orchestration.py    # provider selection precision

# Needs the GPU (declaration uses the LLM):
uv run python scripts/evaluate_declaration.py       # field completeness
uv run python scripts/evaluate_agents.py            # all three → data/evaluation_results.csv
```

Run the test suite:
```bash
uv run pytest
```

---

## Evaluation

Agents are evaluated **in isolation** (each fed gold inputs, so errors don't cascade)
against a unified Golden Dataset: `data/golden_dataset_full.json` — 9 cases, 3 per family
(water_damage, fire, theft), each carrying the expected input/output for every agent. Each
case uses a fixed `today_override` for reproducibility.

| Agent | Metric | Model needed |
|---|---|---|
| **Declaration** | Field completeness (presence of date / incident_type / description / has_photos) | Yes (LLM) |
| **Validation** | Verdict accuracy + correct contract facts (ceiling / deductible) | No (rule-based) |
| **Orchestration** | Provider-selection precision (plombier / expert / serrurier) | No (rule-based) |
| **Expertise** | — not evaluated (assessed manually by domain experts) | — |

`scripts/evaluate_agents.py` consolidates per-case metrics into
`data/evaluation_results.csv` (columns: completeness, validation status/coverage
correctness, provider precision) and prints the aggregates. Dedicated single-agent scripts
(`evaluate_declaration.py`, `evaluate_validation.py`, `evaluate_orchestration.py`) run each
metric on its own.

---

## Data

| Source | Usage |
|---|---|
| `Garanties.md` (contract) | Ceilings, deductibles, declaration deadlines — encoded in `src/agents/validation/rules.py` |
| `Processus.md` (process) | Per-family claims process + service providers — encoded in `PROVIDER_BY_TYPE` (orchestrator) |
| `golden_dataset_full.json` | Evaluation reference (9 cases, 3 per family) |
| `attachments/` | Claim photos analyzed by the VLM (validation coherence + expertise severity) |

These contract/process docs are the authoritative source for the rules; they are not loaded
at runtime — the values are hard-coded into the agents' rule modules.

---

## Project Structure

```
src/
├── config.py
├── inference.py        # UnifiedInference — shared LLM + VLM backend
├── main.py
├── examples.py
├── agents/
│   ├── declaration/
│   │   ├── state.py
│   │   ├── tools.py
│   │   ├── prompts.py
│   │   ├── inference.py
│   │   └── agent.py
│   ├── validation/
│   │   ├── state.py
│   │   ├── rules.py
│   │   ├── vlm_inference.py
│   │   └── agent.py
│   ├── expertise/
│   │   ├── state.py
│   │   ├── rules.py
│   │   ├── prompts.py
│   │   └── agent.py
│   └── orchestrator/
│       ├── state.py
│       ├── handoff.py
│       └── agent.py
└── eval/
    ├── metrics.py
    ├── harness.py
    └── report.py
scripts/
├── evaluate_declaration.py
├── evaluate_validation.py
├── evaluate_orchestration.py
└── evaluate_agents.py
tests/
├── agents/
│   ├── declaration/    # test_agent, test_nodes, test_tools, test_inference
│   ├── validation/     # test_agent, test_rules, test_vlm_inference
│   └── expertise/      # test_agent, test_rules, test_vlm_inference
├── test_inference.py
├── test_orchestrator.py
├── test_eval_metrics.py
└── test_eval_dataset.py
```

## Tech Stack

- **Orchestration**: LangGraph parent pipeline (`src/agents/orchestrator/`) with conditional
  routing and `interrupt()`-based human-in-the-loop gates
- **LLM**: HuggingFace transformers (Qwen2.5-7B-Instruct) — tool calling + follow-up generation
- **VLM**: HuggingFace transformers (Qwen2.5-VL-7B-Instruct) — photo coherence (validation) + damage severity (expertise)
- **Testing**: pytest

---

## Future work

- **Observability** — integrate [Langfuse](https://langfuse.com) to trace each graph run
  (per-node latency, token usage, gate interrupts/resumes, agent inputs/outputs). LangGraph
  has a native Langfuse callback handler, so the pipeline could be instrumented without
  touching agent logic — valuable for debugging and for monitoring SLA/processing-time
  constraints in production.

---

## Step 1 — Declaration Agent

**Goal**: Collect all required information to open a claim file.

### Agent Flow

The agent converses with the policyholder and ensures the following elements are present:

| Element | Description |
|---|---|
| **Date** | Date the incident occurred |
| **Incident type** | Type of event (water damage, fire, burglary…) |
| **Description** | Details of the observed damage |
| **Photos** | Supporting attachments |

> The agent does not validate the information — it only checks for its presence and prompts the policyholder for anything missing.

### Incoming declaration examples

The simulated scenarios live in [`src/examples.py`](src/examples.py). Dates are computed
relative to the run date (e.g. "hier soir"), and attachment filenames refer to real images
in [`data/attachments/`](data/attachments/).

**Example 1** — Water damage (date provided on a 2nd turn)
```
Bonjour,

Il y a eu une fuite dans ma cuisine hier soir à cause de mon voisin du dessus.
Son lave-vaisselle a été mal installé et du coup, le mur est infiltré d'eau et
la peinture se détache (ci-joint une photo).

Cordialement.

[Pièce jointe : WaterDamage_100.jpg]
```

**Example 2** — Burglary (no photos available → rejected at conformity)
```
Bonjour, on m'a cambriolé ce matin, les voleurs sont passés par le vélux de la
chambre et ont volé tous les appareils électroniques. Merci de me contacter
rapidement.
```

**Example 3** — Fire (complete in one turn)
```
Bonjour,

Le <date>, un feu s'est déclaré dans la chambre à cause d'un appareil défectueux,
et a endommagé une grande partie de la pièce. Je souhaiterai être indemnisé pour
pouvoir effectuer les travaux nécessaires.

Bien cordialement.

[Pièce jointe : FireDamage_45.jpg]
[Pièce jointe : FireDamage_31.jpg]
```

---

## Step 2 — Validation Agent

**Goal**: Verify that the declared claim is covered by the policy and that all contractual obligations are met.

### Agent Flow

Receives `final_claim` from the Declaration Agent and runs three sequential checks:

| Check | Method | Description |
|---|---|---|
| **Conformity** | Rule-based | All 4 fields present, photos provided |
| **Coverage** | Rule-based | Incident type covered by contract, declaration deadline respected |
| **Photo coherence** | VLM (Qwen2.5-VL) | Majority of attached photos match the declared incident type |

Each check short-circuits on failure — coverage is not evaluated if conformity fails, photo check is skipped if coverage fails.

Outputs a structured verdict:

```python
{
  "status": "approved" | "rejected",
  "reason": str,        # French message for the policyholder if rejected
  "coverage": {         # None if rejected
    "ceiling": int,     # €
    "deductible": int   # €
  },
  "claim": dict         # original final_claim
}
```

### Coverage rules (from contract)

| Incident type | Ceiling | Deductible | Declaration deadline |
|---|---|---|---|
| Water damage | 25 000 € | 150 € | 5 business days |
| Fire / explosion | 100 000 € | 300 € | 5 business days |
| Theft / vandalism | 20 000 € | 200 € | 2 business days |

---

## Step 3 — Expertise Agent

**Goal**: Produce a structured cost estimate and advisor report for each approved claim.

### Agent Flow

Receives the `verdict` from the Validation Agent (status `approved`) and runs four sequential steps:

| Step | Method | Description |
|---|---|---|
| **assess_severity** | VLM (Qwen2.5-VL) | Majority vote across photos → severity `low / medium / high` |
| **estimate_costs** | Rule-based | Lookup table `{incident_type × severity}` → cost range; apply deductible + ceiling |
| **generate_report** | LLM (Qwen2.5-7B) | Generates French narrative summary for the insurance advisor |
| **finalize** | — | Assembles the `ExpertiseReport` dict |

No final decision is issued to the policyholder. The agent always delegates to an advisor.

Outputs a structured report:

```python
{
  "severity": "low" | "medium" | "high" | "unknown",
  "cost_range": (int, int),          # estimated damage in €
  "compensable_amount": (int, int),  # after deductible and ceiling
  "deductible_applied": int,
  "ceiling_applied": int,
  "summary": str,                    # LLM-generated narrative for advisor
  "claim": dict                      # original claim passthrough
}
```

### Cost estimation rules (indicative — to be validated by domain experts)

| Incident type | Low | Medium | High |
|---|---|---|---|
| Water damage | 200€ – 1 500€ | 1 500€ – 8 000€ | 8 000€ – 20 000€ |
| Fire / explosion | 1 000€ – 10 000€ | 10 000€ – 40 000€ | 40 000€ – 90 000€ |
| Theft / vandalism | 200€ – 2 000€ | 2 000€ – 8 000€ | 8 000€ – 18 000€ |
