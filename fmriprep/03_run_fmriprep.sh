#!/bin/bash
# Run fMRIPrep, one container per participant.
#   bash fmriprep/03_run_fmriprep.sh sci02 hc23
#   PARALLEL=3 NPROCS=4 bash fmriprep/03_run_fmriprep.sh sci01 sci03
#
# --fs-no-reconall                        no FreeSurfer surfaces (volume analysis)
# --output-spaces MNI152NLin6Asym:res-2   the 91x109x91 grid of the atlas
# --slice-time-ref 0                      slice times from the BOLD sidecars (seq+z),
#                                         aligned to the start of each TR
# --derivatives prev=/prev                reuse earlier outputs in OUT (T1 processing,
#                                         BOLD reference, motion and BOLD-to-T1
#                                         transforms); these do not depend on slice timing
set -euo pipefail

IMAGE="nipreps/fmriprep:25.2.6"
BIDS="${BIDS:-/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/01_RAW/bids}"
OUT="${OUT:-/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/04_DERIVATIVES/01_PREPROC/pipeline-fmriprep/v001}"
WORK="${WORK:-/Volumes/WDPassport2/fmriprep_work}"
LICENSE="${FS_LICENSE:-$HOME/fmriprep_work/license.txt}"
LOGS="$(cd "$(dirname "$0")" && pwd)/logs"

PARALLEL="${PARALLEL:-2}"
NPROCS="${NPROCS:-6}"
OMP="${OMP:-3}"
MEM_MB="${MEM_MB:-20000}"

[ "$#" -ge 1 ] || { echo "usage: $0 <label> [label ...]" >&2; exit 1; }
[ -d "$BIDS" ] || { echo "BIDS folder not found; is the drive mounted?" >&2; exit 1; }
[ -f "$LICENSE" ] || { echo "FreeSurfer licence not found: $LICENSE" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker is not running" >&2; exit 1; }
mkdir -p "$OUT" "$LOGS"

run_one() {
  local sub="$1" log="$LOGS/sub-$1.log" wdir="$WORK/sub-$1"
  mkdir -p "$wdir"
  echo "[$(date '+%F %T')] start sub-$sub" | tee -a "$log"
  local t0=$(date +%s)
  docker run --rm --platform linux/amd64 --name "fmriprep_$sub" \
    -v "$BIDS":/data:ro -v "$OUT":/out -v "$OUT":/prev:ro -v "$wdir":/work \
    -v "$LICENSE":/opt/freesurfer/license.txt:ro \
    "$IMAGE" /data /out participant \
      --participant-label "$sub" \
      --fs-no-reconall \
      --output-spaces MNI152NLin6Asym:res-2 \
      --slice-time-ref 0 \
      --derivatives prev=/prev \
      --nprocs "$NPROCS" --omp-nthreads "$OMP" --mem-mb "$MEM_MB" \
      --work-dir /work --notrack --stop-on-first-crash \
      >> "$log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] end sub-$sub exit=$rc after $(( ($(date +%s) - t0) / 60 )) min" | tee -a "$log"
  return $rc
}
export -f run_one
export IMAGE BIDS OUT WORK LICENSE LOGS NPROCS OMP MEM_MB

printf '%s\n' "$@" | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {} || true
