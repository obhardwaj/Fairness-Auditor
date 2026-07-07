#!/bin/sh
# Blocks until Postgres is accepting connections, then execs the given command.
#
# Usage (in docker-compose.yml):
#   command: ["./docker/wait-for-postgres.sh", "postgres", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
#
# This exists because Compose's `depends_on: condition: service_healthy` (already
# used in docker-compose.yml) covers most cases, but this script is a cheap
# extra safety net for local dev when healthchecks are slow to register, and
# it's a more portable pattern if this ever runs outside Compose (e.g. k8s
# initContainer replaced by this, or a plain `docker run`).

set -e

host="$1"
shift
cmd="$@"

until pg_isready -h "$host" -U "${POSTGRES_USER:-fairness_user}" > /dev/null 2>&1; do
  echo "Postgres is unavailable at host '$host' - sleeping"
  sleep 1
done

echo "Postgres is up - executing command"
exec $cmd
