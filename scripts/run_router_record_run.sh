#!/usr/bin/env bash
#
# One command to finish ticket 24: bring the stack up, capture the eval-set
# Record run for both Groq rungs (rotating keys and pacing itself through the
# free tier's per-minute and per-day limits), commit as it goes, then drive
# the router ablation.
#
# Run it and leave it — days of wall-clock, but no attention needed. It logs to
# record_run.log and survives anything short of the machine going down; if it
# does stop, just run it again and it resumes.
#
#   bash scripts/run_router_record_run.sh
#
# Deliberately NOT `set -e`: a days-long loop must not die on one transient
# hiccup (a suspended laptop interrupting `sleep`, a flaky API call, a compose
# restart). Every failure that is not "done" just waits and retries.

set -uo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

LOG="record_run.log"
SLEEP_ON_CAP="${SLEEP_ON_CAP:-3600}"   # wait after every key is capped
SLEEP_ON_ERROR="${SLEEP_ON_ERROR:-120}" # wait after an unexpected failure
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "==> run started $(date)"

stack_up() {
  docker compose up -d --wait api qdrant postgres redis ai >/dev/null 2>&1 \
    || docker compose up -d >/dev/null 2>&1 || true
}

refresh_token() {
  local token
  token=$(docker compose logs migrate 2>/dev/null \
    | grep -B1 'Deflection assistant' \
    | grep -o 'nvk_live_[A-Za-z0-9._-]*' \
    | tail -1) || true
  if [ -n "${token:-}" ]; then
    if grep -q '^NIVARA_ASSISTANT_TOKEN=' .env; then
      sed -i "s|^NIVARA_ASSISTANT_TOKEN=.*|NIVARA_ASSISTANT_TOKEN=$token|" .env
    else
      printf '\nNIVARA_ASSISTANT_TOKEN=%s\n' "$token" >> .env
    fi
  fi
}

commit_recordings() {
  git add recordings/ >/dev/null 2>&1 || true
  git commit -q -m "chore: eval-set Record run for the router ablation" >/dev/null 2>&1 || true
}

load_env() {
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  export NIVARA_MODEL_TRANSPORT=live
}

echo "==> bringing the stack up"
stack_up
refresh_token
load_env

echo "==> indexing the corpus"
python scripts/index_corpus.py || true

echo "==> recording (rungs: groq-gpt-oss-120b, groq-gpt-oss-20b)"
while true; do
  stack_up
  refresh_token
  load_env

  python scripts/record_eval.py --slice all \
    --rung groq-gpt-oss-120b --rung groq-gpt-oss-20b
  rc=$?

  commit_recordings

  if [ "$rc" -eq 0 ]; then
    echo "==> recording complete"
    break
  elif [ "$rc" -eq 2 ]; then
    echo "!! record_eval.py exited 2 (configuration) — check .env; stopping" >&2
    exit 2
  else
    # 3 = every key capped for the day; anything else = a transient failure.
    wait=$SLEEP_ON_CAP
    [ "$rc" -ne 3 ] && wait=$SLEEP_ON_ERROR
    echo "==> record_eval.py exited $rc; sleeping ${wait}s then resuming ($(date))"
    sleep "$wait" || true
  fi
done

echo "==> driving the router ablation (no quota)"
unset NIVARA_MODEL_TRANSPORT
python scripts/router_ablation.py --drive
git add recordings/ eval/router_ablation.json eval/router_ablation.md >/dev/null 2>&1 || true
git commit -q -m "feat: the router ablation, driven from the Record run" >/dev/null 2>&1 || true

cat <<'EOF'

============================================================
Record run + ablation done. eval/router_ablation.md now has the table.
============================================================
EOF
