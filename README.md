<<<<<<< HEAD
# Fairness-Auditor
Not just "here are your fairness metrics" — an agent that reasons about which metrics are even valid for your data, flags contradictions between them, and tests mitigation trade-offs empirically.


An agentic platform that audits trained classifiers for statistical fairness violations,
quantifies uncertainty in fairness metrics, tests mitigation strategies, and produces a
plain-language report — orchestrated with LangGraph, computed with scikit-learn/Fairlearn/AIF360.

## Project layout

```
fairness-auditor/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI route handlers
│   │   ├── core/         # config, db session, celery/background job setup
│   │   ├── models/        # SQLAlchemy ORM models
│   │   ├── schemas/       # Pydantic request/response schemas
│   │   ├── ml/            # metrics, mitigation, data profiling (pure DS code, no LLM)
│   │   ├── agents/        # LangGraph state graph + node definitions
│   │   └── main.py        # FastAPI app entrypoint
│   ├── alembic/           # DB migrations
│   ├── tests/             # pytest, incl. regression tests vs published COMPAS numbers
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── hooks/
│   └── Dockerfile
├── docker/
├── docker-compose.yml
└── .env.example
```

## Week 1 goals (this scaffold covers the skeleton for)
1. Docker Compose stack: api, postgres, redis, frontend
2. Postgres schema: datasets, models, audit_runs, metric_results, mitigation_results
3. Data ingestion + protected-attribute/proxy detection stub
4. Baseline model training stub (logistic regression, gradient boosting)

See `backend/app/ml/` and `backend/app/models/` to start filling in Week 1 logic.

## Running locally

```bash
cp .env.example .env
docker compose up --build
```

API docs: http://localhost:8000/docs
Frontend: http://localhost:5173
