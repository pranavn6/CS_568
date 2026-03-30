# Multi-Agent Reasoning Module

## Overview

This module implements a multi-agent reasoning system for analyzing counterfactual public health scenarios.
Instead of generating a single prediction, the system produces multiple perspectives from specialized agents and synthesizes them into a final recommendation.

The goal is to make trade-offs explicit across public health, healthcare capacity, and economic impact.

---

## System Architecture

The system takes structured simulation output as input and produces agent-based analyses as output.

### Input

```json
{
  "interventions": {
    "mask_mandate": true,
    "school_closure": false,
    "vaccination_rate": 0.6
  },
  "metrics": {
    "cases": [...],
    "hospital_load": [...],
    "deaths": [...],
    "economic_cost": 0.42
  }
}
```

### Output

```json
{
  "epidemiologist": "...",
  "healthcare": "...",
  "economist": "...",
  "consensus": "..."
}
```

Each agent produces concise, structured reasoning grounded in the provided data.

---

## Agents

### Epidemiologist

* Focus: infection spread, cases, deaths
* Evaluates whether interventions are sufficient
* Identifies public health risks
* Does not discuss economic or operational factors

### Healthcare Capacity Analyst

* Focus: hospital load and system strain
* Assesses risk of overload
* Identifies operational constraints
* Does not analyze economic trade-offs

### Economic Impact Analyst

* Focus: economic disruption and societal cost
* Evaluates trade-offs between restrictions and stability
* Uses health trends only as context for economic decisions

### Consensus Agent

* Synthesizes all agent responses
* Identifies agreement and key tensions
* Produces a balanced recommendation

---

## How It Works

1. Load a structured simulation scenario
2. Run each agent independently using prompt-based reasoning
3. Generate outputs from:

   * Epidemiologist
   * Healthcare Analyst
   * Economic Analyst
4. Pass all outputs into the consensus agent
5. Save results as structured JSON

---

## Setup

### Install dependencies

```bash
python -m pip install openai
```

### Set API key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## Running the System

### Run a single scenario

```bash
python agent_runner.py
```

### Run all test scenarios

```bash
python test_runner.py
```

Outputs are saved in:

```
outputs/
```

---

## Example Scenarios

Located in:

```
data/
```

Includes:

* Moderate intervention scenario
* High-risk scenario (healthcare overload)
* Controlled scenario (high restrictions, high cost)

---

## Design Principles

* Prompt-based agents (no model training required)
* Strict role separation to avoid overlapping reasoning
* Fixed-length outputs for clarity and comparability
* Structured JSON outputs for evaluation and integration

---

## Current Status

* Multi-agent pipeline implemented
* LLM integration complete
* Prompt refinement completed
* Clear role separation achieved
* Outputs saved for evaluation

---

## Next Steps

* Add cross-agent critique (optional extension)
* Integrate with simulation module
* Add logging for evaluation metrics
* Connect to frontend interface
