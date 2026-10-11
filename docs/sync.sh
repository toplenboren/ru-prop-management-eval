#!/bin/sh
# Copies eval results into docs/data so the site is self-contained for static hosting.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$(cd "$(dirname "$0")" && pwd)/data"
mkdir -p "$D"
cp "$ROOT/results/grades.tsv" "$ROOT/data/dataset.tsv" "$ROOT/llms_config.jsonl" "$D/"
cp "$ROOT"/results/*_answers.jsonl "$ROOT"/results/*_jev_grades.jsonl "$D/"
ls "$D" | grep '_answers\.jsonl$' | sed 's/_answers\.jsonl$//' \
  | python3 -c 'import sys,json;print(json.dumps([l.strip() for l in sys.stdin if l.strip()],indent=0))' \
  > "$D/manifest.json"
echo "Synced $(ls "$D" | wc -l | tr -d ' ') files to $D"
