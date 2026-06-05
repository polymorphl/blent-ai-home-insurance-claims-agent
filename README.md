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

Agents are orchestrated via a multi-agent system or a purpose-built workflow.

See [GRAPHS.md](GRAPHS.md) for Mermaid diagrams of each agent's internal flow.

---

## Constraints

- **Data stays on-premise** — no calls to OpenAI, Anthropic, or any external API; all inference runs locally
- **Open-weight models only** — currently Qwen2.5-7B-Instruct for text (tool calling + generation); a VLM will be required for the Expertise Agent (image analysis)
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

Run 6 simulated examples (Declaration → Validation → Expertise pipeline):
```bash
uv run python -m src.main
```

Run Declaration Agent evaluation:
```bash
uv run python scripts/evaluate_declaration.py
```

Run Validation Agent evaluation:
```bash
uv run python scripts/evaluate_validation.py
```

---

## Evaluation

Agents are evaluated against Golden Datasets built for this project. No threshold is defined yet — metrics serve as a baseline for discussion.

| Agent | Dataset | Metric |
|---|---|---|
| **Declaration Agent** | `data/golden_dataset.json` (3 cases) | Completeness of collected fields |
| **Validation Agent** | `data/golden_dataset_validation.json` (6 cases) | Accuracy of verdict (approved / rejected) |
| **Expertise Agent** | — | No automated evaluation — assessed manually by domain experts |

Each dataset uses fixed reference dates (`today_override`) to ensure reproducibility regardless of when evaluation is run.

---

## Data

| Source | Usage |
|---|---|
| Claims process documentation | Reference for expected steps, insurer obligations, case closure rules |
| Policyholder's home insurance policy | Reference for coverage validation against the declared claim |

---

## Project Structure

```
src/
├── config.py
├── main.py
├── examples.py
└── agents/
    ├── declaration/
    │   ├── state.py
    │   ├── tools.py
    │   ├── prompts.py
    │   ├── inference.py
    │   └── agent.py
    ├── validation/
    │   ├── state.py
    │   ├── rules.py
    │   ├── vlm_inference.py
    │   └── agent.py
    └── expertise/
        ├── state.py
        ├── rules.py
        ├── prompts.py
        └── agent.py
scripts/
├── evaluate_declaration.py
└── evaluate_validation.py
```

## Tech Stack

- **Orchestration**: LangGraph (multi-agent workflow)
- **LLM**: HuggingFace transformers (Qwen2.5-7B-Instruct) — tool calling + follow-up generation
- **VLM**: HuggingFace transformers (Qwen2.5-VL-7B-Instruct) — photo coherence check
- **Testing**: pytest

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

**Example 1** — Water damage (missing date)
```
Hello,
There was a leak in my kitchen last night due to my upstairs neighbor.
His dishwasher was poorly installed, and as a result, the wall is soaked
and the paint is peeling off (photo attached).
Regards.
[Attachment: IMG_4580.jpg]
```

**Example 2** — Burglary (missing date and photos)
```
Hello, I was burgled this morning. The thieves came in through the bedroom
skylight and stole all the electronics. Please contact me as soon as possible.
```

**Example 3** — Fire (complete)
```
Hello,
On 10/09/2025, a fire broke out in the bedroom due to a faulty appliance
and damaged a large part of the room. I would like to be compensated to
carry out the necessary repairs.
Kind regards.
[Attachment: Chambre_1.jpg]
[Attachment: Chambre_2.jpg]
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
