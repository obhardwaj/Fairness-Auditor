# Bias & Fairness Auditor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

An agentic platform that audits trained classifiers for statistical fairness
violations. It doesn't just compute fairness metrics — it reasons about which
metrics are even appropriate given the data, quantifies uncertainty with
bootstrap confidence intervals, empirically tests three mitigation strategies
across the pre/in/post-processing spectrum, and produces a plain-language
report — all orchestrated with LangGraph and grounded in Fairlearn/AIF360.

---

## Problem Statement

Fairness metrics are not a checklist you run once — they're a set of
competing, formally incompatible objectives. A classifier cannot generally
satisfy demographic parity, equalized odds, and calibration simultaneously
unless the underlying base rates are equal across groups (the fairness
**impossibility theorem**). Most tooling in this space stops at reporting
metrics; it doesn't reason about *which* metric is appropriate for a given
context, doesn't quantify whether an observed disparity is statistically
robust or a sampling artifact, and doesn't compare mitigation strategies
against each other with the same rigor.

This project builds that missing layer: an agentic audit pipeline that
profiles a dataset, computes a full fairness metric suite with bootstrap
confidence intervals, critiques its own results for contradictions, selects
and applies a mitigation strategy based on evidence, and explains all of it
in plain language for a non-technical stakeholder.

## Dataset

[ProPublica's COMPAS recidivism dataset](https://github.com/propublica/compas-analysis)
(`compas-scores-two-years.csv`) — the canonical benchmark in the algorithmic
fairness literature, chosen specifically so results are directly comparable
to published analysis. Standard ProPublica filtering is applied (dropping
rows with an unreliable screening window, unknown recidivism outcome, or
ordinary traffic offenses), reducing the raw file to **6,172 rows** — matching
the widely-cited filtered count used across the literature, which served as
an early validation check that the pipeline's filtering logic was correct.

For binary-group mitigation methods (AIF360's Reweighing requires exactly two
groups), the dataset is further restricted to African-American vs. Caucasian
defendants — the two largest groups (3,696 and 2,454 rows) and the pairing
ProPublica's own headline analysis focused on.



## Project Structure

```
fairness-auditor/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── audits.py          # audit run CRUD, list, report, calibration, mitigation-comparison
│   │   │   └── datasets.py        # dataset ingestion, COMPAS baseline training, model listing
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic settings (DB, Redis, Groq API key)
│   │   │   ├── db.py              # SQLAlchemy engine/session
│   │   │   ├── celery_app.py      # Celery app config
│   │   │   └── tasks.py           # async entrypoint -> app.agents.graph.run_audit_graph
│   │   ├── models/
│   │   │   └── models.py          # Dataset, MLModel, AuditRun, MetricResult, MitigationResult
│   │   ├── ml/                    # pure DS code, no LLM dependency
│   │   │   ├── profiling.py       # protected attribute + proxy detection
│   │   │   ├── compas_pipeline.py # load/filter COMPAS, feature matrix, baseline training
│   │   │   ├── baseline_models.py # logistic regression / gradient boosting training
│   │   │   ├── metrics.py         # fairness metric suite, bootstrap CIs, calibration curves
│   │   │   └── mitigation.py      # reweighing, ExponentiatedGradient, ThresholdOptimizer
│   │   ├── agents/
│   │   │   ├── llm.py             # shared Groq client
│   │   │   └── graph.py           # LangGraph state graph: profiler -> metric_runner -> critic -> mitigation -> report
│   │   ├── schemas/
│   │   │   └── schemas.py         # Pydantic request/response schemas
│   │   └── main.py                # FastAPI app entrypoint
│   ├── scripts/
│   │   └── persist_mitigation_comparison.py  # one-off: persists all 8 baseline+mitigation rows
│   ├── alembic/                   # DB migrations
│   ├── tests/
│   │   ├── test_profiling.py      # protected attribute / proxy detection tests
│   │   └── test_metrics.py        # regression test vs. published ProPublica COMPAS numbers
│   ├── data/                      # compas-scores-two-years.csv (gitignored) + artifacts/ (trained models, gitignored)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ParetoChart.jsx        # accuracy vs. disparate impact scatter, all 8 method/algorithm points
│   │   │   ├── AuditReport.jsx        # report text, profiler notes, critic contradictions, chosen mitigation — polls until completed
│   │   │   ├── AuditRunSelector.jsx   # dropdown of all audit runs, polls for status updates
│   │   │   └── CalibrationChart.jsx   # per-group calibration curves, computed on demand
│   │   │   ├── RunAuditButton.jsx     # triggers a new audit run from the UI, no terminal needed
│   │   │   └── StatusBox.jsx          # shared loading skeleton / error box components
│   │   └── App.jsx
│   └── Dockerfile
├── docker/
│   ├── postgres/init.sql          # enables uuid-ossp, pg_stat_statements on first boot
│   └── wait-for-postgres.sh       # used as api/worker's container command
├── docker-compose.yml
├── .env.example
├── .gitignore
├── LICENSE
└── backend/.dockerignore, frontend/.dockerignore
```

## Architecture

               ┌─────────────┐

FastAPI ──────► │ Postgres │ ◄────── Celery worker
(REST API) │ (audit data)│ (async LangGraph runs)
└─────────────┘
▲
│
┌──────┴──────┐
│ LangGraph │
│ agent graph │
└──────┬──────┘
┌──────────┬─────────┼──────────┬───────────┐
▼ ▼ ▼ ▼ ▼
profiler → metric_runner → critic → [mitigation] → report
│ │ │ │ │
LLM Fairlearn LLM Fairlearn/ LLM
(Groq) + bootstrap (Groq) AIF360 (Groq)


**Design boundary, held throughout:** LLM nodes reason about *which* metrics
or mitigations are appropriate and *why* — they never compute fairness math
themselves. Every number in a report traces back to Fairlearn, AIF360, or
scipy/numpy, not to model-generated arithmetic.

The dashboard is fully self-service: new audit runs can be triggered, tracked,
and reviewed entirely from the UI (`http://localhost:5173`) — no terminal or
API client required after initial setup.


- **`profiler`** — computes ground-truth base rate gaps between groups, then
  reasons about whether demographic parity or equalized odds is the more
  appropriate target given that gap.
- **`metric_runner`** — computes demographic parity difference, disparate
  impact ratio, equalized odds difference, FPR/FNR differences, and
  per-group calibration curves, each with a bootstrap 95% confidence
  interval (1000 resamples in the core suite, 500 in agent runs for
  latency).
- **`critic`** — given the metrics with their CIs, flags contradictions
  (e.g. a metric passing while a related one fails) and explicitly invokes
  the impossibility theorem where relevant.
- **`mitigation`** *(conditional — only runs if disparate impact ratio 
  0.8, the legal "80% rule" threshold)* — selects one of three methods based
  on the critic's findings and prior empirical results, applies it, and
  recomputes the full metric suite on the mitigated model.
- **`report`** — synthesizes the full run into a plain-language report for a
  non-technical stakeholder.


## Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI, SQLAlchemy, Alembic |
| Database | PostgreSQL |
| Async jobs | Celery + Redis |
| ML / fairness | scikit-learn, Fairlearn, AIF360, scipy (bootstrap CIs) |
| Agent orchestration | LangChain, LangGraph, Groq (Llama 3.3 70B) |
| Frontend | React, Vite, Tailwind, Recharts |
| Infra | Docker Compose |

## Methodology

### Metric suite
All metrics are computed via Fairlearn's `MetricFrame`/reduction utilities —
not reimplemented by hand — and validated against ProPublica's own published
COMPAS analysis as a regression test (`tests/test_metrics.py`) before being
trusted on this project's own trained models. Every metric is reported with
a bootstrap 95% confidence interval (percentile method, 500–1000 resamples)
rather than a bare point estimate, since a single number cannot distinguish
a genuine disparity from a sampling artifact.

### Mitigation methods compared
Three methods spanning the full pre/in/post-processing spectrum:

| Method | Stage | Mechanism |
|---|---|---|
| Reweighing (AIF360) | Pre-processing | Adjusts per-sample training weights so protected attribute and label become statistically independent |
| ExponentiatedGradient (Fairlearn) | In-processing | Directly constrains the training objective to a fairness constraint |
| ThresholdOptimizer (Fairlearn) | Post-processing | Adjusts decision thresholds per group after training |

### Reproducibility note
Both `ExponentiatedGradient` and `ThresholdOptimizer` have stochastic
elements in their default `.predict()` behavior (a randomized sample from a
classifier mixture, and randomized threshold interpolation, respectively).
Two corrections were applied so results are reproducible run-to-run:
- **ExponentiatedGradient**: metrics are computed from Fairlearn's internal
  `_pmf_predict` (the deterministic weighted-average probability across the
  mixture) thresholded at 0.5, rather than a single stochastic `.predict()`
  sample. This is a documented approximation, not identical to the exact
  classifier the formal fairness guarantee covers — the guarantee applies to
  the true randomized classifier, not this deterministic proxy.
- **ThresholdOptimizer**: `np.random.seed(42)` is set immediately before
  each `.predict()` call, verified to produce byte-identical output across
  repeated runs.

## Results

All results below are on the race-binary (African-American vs. Caucasian)
subset, for both a logistic regression and a gradient boosting baseline
classifier trained on `two_year_recid`.

| Method | LR Disparate Impact | LR Accuracy | GB Disparate Impact | GB Accuracy |
|---|---|---|---|---|
| Baseline | 0.480 | 0.676 | 0.523 | 0.697 |
| Reweighing (pre) | 0.876 | 0.649 | 0.754 | 0.662 |
| ExponentiatedGradient (in)* | 0.602 | 0.640 | 0.947 | 0.665 |
| ThresholdOptimizer (post) | **0.966** | 0.637 | **0.964** | 0.641 |

*See reproducibility note above regarding the PMF-thresholded evaluation.

**Key finding:** ThresholdOptimizer is both the strongest performer and the
most consistent across model architectures (0.966 vs. 0.964 — nearly
identical), while the other two methods are architecture-dependent:
reweighing works better for logistic regression, ExponentiatedGradient works
far better for gradient boosting. This suggests **the choice of *when* in
the pipeline to intervene (pre/in/post-processing) matters more for
reliability than which specific base model is being debiased** — a pattern
consistent with the mechanical intuition that post-processing methods act
directly on the decision boundary, while pre/in-processing methods leave
more flexible model classes room to partially reconstruct the original
disparity through feature interactions.

All three methods clear the legal 80% disparate impact threshold at a
roughly similar accuracy cost (3–5 percentage points), illustrating that a
meaningful fairness improvement does not necessarily require a large
accuracy sacrifice — the choice of *method*, not just the accuracy/fairness
tradeoff itself, is what determines the outcome.

## Limitations

- **PMF-thresholding for ExponentiatedGradient** is a deterministic
  approximation of a stochastic classifier — see the reproducibility note
  above. The formal fairness guarantee applies to the true randomized
  classifier, not this proxy.
- **Calibration is not reported for ThresholdOptimizer**, which by design
  outputs group-specific hard decisions rather than probability scores —
  calibration is not a meaningful concept for this method's output.
- **Single dataset**: results are demonstrated on COMPAS only. The framework
  is designed to generalize (see `app/ml/profiling.py`'s dataset-agnostic
  protected-attribute/proxy detection), but has not yet been validated on a
  second dataset from a different domain.
- **Binary-group restriction**: AIF360's Reweighing implementation as used
  here assumes exactly two groups; the full 6-category race breakdown in
  COMPAS is not exercised by the mitigation comparison.
- **LLM agent nondeterminism**: profiler/critic/mitigation reasoning is
  generated by an LLM (Groq Llama 3.3 70B) and is not guaranteed to be
  identical across runs, even though the underlying metrics it reasons over
  are fully reproducible.

## Local Development

```bash
git clone https://github.com/obhardwaj/Fairness-Auditor.git
cd fairness-auditor
cp .env.example .env        # fill in GROQ_API_KEY
docker compose up --build
```

- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:5173`

Drop `compas-scores-two-years.csv` (from
[ProPublica's repo](https://github.com/propublica/compas-analysis)) into
`backend/data/` before running the ingestion/training endpoints — the
dataset is not committed to this repository.

```bash
# generate DB tables
docker compose exec api alembic upgrade head

# ingest + train baseline models
curl -X POST "http://localhost:8000/datasets/train-compas-baseline"

# run the full agentic audit (async via Celery) -- either via curl:
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"model_id": "<uuid from the models table>"}'

# ...or from the dashboard itself (http://localhost:5173): pick a model
# from the "Run New Audit" dropdown and click Run -- the audit run selector,
# report, and calibration chart all poll automatically until it completes.

## Project Structure

See inline comments in `backend/app/` — organized by concern:
`api/` (routes), `ml/` (metrics, mitigation, profiling — no LLM dependency),
`agents/` (LangGraph state graph), `models/` (SQLAlchemy schema).