#!/bin/bash
# Wait for fMRIPrep to finish, then run QC, the MATLAB input preparation and
# the MATLAB CVR/BAT step for all participants.
#   nohup bash fmriprep/07_after_fmriprep.sh > fmriprep/logs/after_fmriprep.log 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$HERE")"
PY="${PYTHON:-python}"
MATLAB="${MATLAB:-matlab}"
log() { echo "[$(date '+%F %T')] $*"; }

while pgrep -f 05_queue_more.sh >/dev/null || [ -n "$(docker ps -q --filter name=fmriprep_)" ]; do
  sleep 120
done
log "fMRIPrep finished"

for f in "$HERE"/logs/sub-*.log; do
  last=$(grep "end sub" "$f" | tail -1)
  [[ "$last" == *exit=0* ]] || log "failed: $(basename "$f" .log): $last"
done

log "QC"
"$PY" "$HERE/04_qc_outputs.py"

log "MATLAB input"
"$PY" "$HERE/06_prepare_dlrs_input.py"
grep -v ",ok$" "$HERE/dlrs_input_manifest.csv"

log "MATLAB"
cd "$PROJ/preprocessing" && "$MATLAB" -batch rs_running_fmriprep > matlab_fmriprep.log 2>&1
grep -E "Successful|Failed|ERROR" matlab_fmriprep.log
log "done"
