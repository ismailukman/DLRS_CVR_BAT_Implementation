#!/bin/bash
# Queue participants, at most MAX fMRIPrep runs at once. Waits until every
# participant in AFTER has started.
#   AFTER="sci03 sci04" bash fmriprep/05_queue_more.sh sci09 sci11
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MAX="${MAX:-3}"
AFTER="${AFTER:-}"

# runs in progress, counted from our own run processes (does not depend on Docker answering)
running() { pgrep -f "03_run_fmriprep.sh" | wc -l | tr -d ' '; }
after_started() {
  for s in $AFTER; do [ -f "$HERE/logs/sub-$s.log" ] || return 1; done
}

echo "[$(date '+%F %T')] queue: $*  (max $MAX; after: $AFTER)"
pids=()
for sub in "$@"; do
  until after_started && [ "$(running)" -lt "$MAX" ]; do sleep 30; done
  echo "[$(date '+%F %T')] starting sub-$sub (running: $(running))"
  PARALLEL=1 NPROCS=4 OMP=2 MEM_MB=14000 bash "$HERE/03_run_fmriprep.sh" "$sub" >/dev/null 2>&1 &
  pids+=($!)
  sleep 10
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date '+%F %T')] queue finished"
