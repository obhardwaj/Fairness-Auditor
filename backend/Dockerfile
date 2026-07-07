FROM python:3.11-slim

WORKDIR /app

# System deps needed by scikit-learn / aif360 / psycopg2, plus postgresql-client
# for pg_isready (used by docker/wait-for-postgres.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
