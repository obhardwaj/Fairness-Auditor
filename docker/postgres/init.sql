-- Runs automatically the first time the postgres container initializes its
-- data volume (Postgres's official image executes any .sql files placed in
-- /docker-entrypoint-initdb.d/ on first boot only — it will NOT re-run this
-- on subsequent restarts unless the pg_data volume is wiped).

-- uuid-ossp gives us uuid_generate_v4() at the DB level. Our SQLAlchemy models
-- currently generate UUIDs in Python (see app/models/models.py: gen_uuid()),
-- so this isn't strictly required yet — but it's useful to have available if
-- we ever want a DB-side default (server_default=text("uuid_generate_v4()"))
-- instead of generating them in the application layer.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- pg_stat_statements is handy during development for spotting slow queries
-- once the metric/mitigation tables start seeing real read/write volume.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
