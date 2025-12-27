# Stable Lasso IV Pipeline

This repository implements a **three-stage pipeline for high-dimensional IV estimation** using stability-based Lasso selection.  
The code is designed as a **modular preprocessing and selection system** that prepares data for downstream IV / 2SLS estimation.

The repository focuses on **architecture and workflow**, not on a specific empirical application.

---

## Repository Architecture

The pipeline is strictly sequential and stateful.  
Each step writes outputs to disk that are required by the next step.

Step 1 → Step 2 → Step 3 → IV estimation (external)

repo/
│
├── step1_select_controls_lasso.py
├── step2_residualize_sample.py
├── step3_select_instruments_belloni_parallel_progress.py
│
├── results/ # machine-readable outputs
├── plots/ # control-selection diagnostics
├── figure/ # IV-selection diagnostics
└── temp/ # scratch space (optional)


---

## Pipeline Logic (High Level)

### Step 1 — Stable Control Selection
**Goal:** identify control variables that are robustly relevant.

- Uses repeated subsampling + Lasso
- Applies *double selection* (outcome and treatment)
- Retains only controls that appear consistently across batches
- Outputs a minimal, stable control set

**Output (required by Step 2):**

results/selected_controls.csv


---

### Step 2 — Residualization Layer
**Goal:** orthogonalize key variables with respect to selected controls.

- Loads selected controls from Step 1
- Residualizes outcome, treatment, and weights
- Produces a compact dataset for IV selection and estimation
- Handles mixed numeric / categorical controls internally

**Output (required by Step 3):**

results/residualized_dataset.csv


---

### Step 3 — Stable Instrument Selection
**Goal:** select instruments from a large candidate set in a disciplined way.

- Uses theoretical (Belloni-type) Lasso penalty
- Runs repeated subsampling
- Executes batches in parallel
- Aggregates selection frequency and strength

**Output (final selection artifacts):**

results/selected_instruments.csv
results/instrument_selection_frequency.csv


---

## Execution Order

The scripts must be run **in order**:

```bash
python step1_select_controls_lasso.py
python step2_residualize_sample.py
python step3_select_instruments_belloni_parallel_progress.py

Each script checks that the required outputs from the previous step exist.

What This Repository Does NOT Do

❌ Final IV / 2SLS estimation

❌ Standard error computation

❌ Inference or hypothesis testing

❌ Application-specific specification choices

Those steps are intentionally left outside the pipeline.
