#!/usr/bin/env bash
# End-to-end smoke test for the Phase 1 inference server.
# Assumes the server is already running (e.g. `./run.sh` in another terminal).
# Usage: ./test.sh [host:port]
set -e
ADDR="${1:-localhost:8080}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== GET /health =="
curl -s -w "\nHTTP %{http_code}\n" "http://$ADDR/health"
echo

for img in dog cat; do
  echo "== POST /predict  (testdata/$img.jpg) =="
  curl -s -X POST "http://$ADDR/predict" \
       --data-binary "@$HERE/testdata/$img.jpg" \
       -w "\nHTTP %{http_code}\n"
  echo
done

echo "== POST /predict  (malformed body -> expect 400) =="
printf 'not an image' | curl -s -X POST "http://$ADDR/predict" \
     --data-binary @- -w "\nHTTP %{http_code}\n"
