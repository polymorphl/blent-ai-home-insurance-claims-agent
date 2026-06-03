# AI Agent - Home Insurance Claims (AssurHabitat)

Multi-agent AI pipeline to automate home insurance claims processing for AssurHabitat.

## Run

Install dependencies:
```bash
uv sync
```

Run 3 simulated examples:
```bash
uv run python -m src.main
```

Run evaluation on golden dataset:
```bash
uv run python scripts/evaluate_declaration.py
```

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

---

## Constraints

- **No third-party APIs** (OpenAI, Anthropic, etc.) — data must not leave the company's infrastructure
- **Open-weight models only** — Mistral, Llama, or equivalent, run locally (applies to both LLMs and VLMs)
- **No UI** — the pipeline runs asynchronously in the background per claim
- **Agent handoff** — each agent must receive all information needed to execute the next step

---

## Evaluation

Agents are evaluated against a Golden Dataset built for this project. No threshold is defined yet — metrics serve as a baseline for discussion.

| Agent | Metric |
|---|---|
| **Declaration Agent** | Completeness of collected information |
| **Validation Agent** | Accuracy of contract fact retrieval relative to the declared claim |
| **Expertise Agent** | No automated evaluation — assessed manually by domain experts |

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
    └── declaration/
        ├── state.py
        ├── tools.py
        ├── prompts.py
        ├── inference.py
        └── agent.py
scripts/
└── evaluate_declaration.py
```

## Tech Stack

- **Orchestration**: LangGraph (multi-agent workflow)
- **LLM**: HuggingFace transformers (Llama 3.1 8B Instruct)
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
