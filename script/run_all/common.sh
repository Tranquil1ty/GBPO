#!/usr/bin/env bash
# Shared serial runner. Invoke run_toys.sh, run_beauty.sh or run_sports.sh.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PATH=/data/share2/project/pengfei/env/gbpo
DATASET=${1:?Missing dataset}
shift
MODES=("$@")
LOG_DIR="$ROOT_DIR/logs/run_all/${DATASET}/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %z')" "$*"; }
CURRENT=initialization
trap 'rc=$?; if (( rc != 0 )); then log "FAILED: $CURRENT (exit=$rc); remaining tasks will not run."; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log "Dataset: $DATASET; queue: ${MODES[*]}"
log "Logs: $LOG_DIR"
# All three dataset queues share the same GPUs. Hold this lock for the whole queue.
CURRENT='waiting for GPU queue lock'
exec 9>"$ROOT_DIR/logs/run_all/.gpu_queue.lock"
log 'Waiting for shared GPU queue lock (other run_all queues must finish first).'
flock -x 9
log 'Acquired GPU queue lock.'

CURRENT='conda activation'
CONDA_EXE_PATH="${CONDA_EXE:-}"
if [[ ! -x "$CONDA_EXE_PATH" ]]; then
    CONDA_EXE_PATH="$(type -P conda || true)"
fi
if [[ ! -x "$CONDA_EXE_PATH" && -x /data/miniconda/bin/conda ]]; then
    CONDA_EXE_PATH=/data/miniconda/bin/conda
fi
if [[ ! -x "$CONDA_EXE_PATH" ]]; then
    log 'ERROR: conda executable not found.'
    exit 1
fi
# Conda activation scripts may reference unset variables.
set +u
eval "$("$CONDA_EXE_PATH" shell.bash hook)"
conda activate "$ENV_PATH"
set -u
log "Conda environment: $CONDA_PREFIX; Python: $(command -v python)"
export PYTHONUNBUFFERED=1
cd "$ROOT_DIR/script"

case "$DATASET" in
    Beauty) OFFSET=0 ;;
    Sports) OFFSET=1 ;;
    Toys) OFFSET=2 ;;
    *) log "Unknown dataset: $DATASET"; exit 2 ;;
esac
for MODE in "${MODES[@]}"; do
    CURRENT="$DATASET/$MODE"
    case "$MODE" in
        SFT|GBPO) BASE_PORT=29501 ;;
        GRPO) BASE_PORT=29504 ;;
        GSPO) BASE_PORT=29507 ;;
        DAPO) BASE_PORT=29510 ;;
        GPPO) BASE_PORT=29513 ;;
        DUAL_PPO) BASE_PORT=29516 ;;
        *) log "Unknown mode: $MODE"; exit 2 ;;
    esac
    PORT=$((BASE_PORT + OFFSET))
    # Match run.sh's original training log directory and naming convention.
    if [[ "$MODE" == SFT ]]; then
        LOG_SUFFIX=free
    else
        LOG_SUFFIX=OffPolicy_beam_topk5_K32_trie_gen
    fi
    TASK_LOG="$ROOT_DIR/logs_rl/$DATASET/main_${MODE}_${LOG_SUFFIX}_$(date +%Y%m%d_%H%M%S)_$$.log"
    mkdir -p "$(dirname -- "$TASK_LOG")"
    START=$SECONDS
    log "START $CURRENT; port=$PORT; log=$TASK_LOG"
    if [[ "$MODE" == SFT ]]; then
        ARGS=("$DATASET" "$MODE" torchrun)
    else
        ARGS=("$DATASET" "$MODE" torchrun false false beam trie_gen)
    fi
    # Wait for training; keep detailed output out of the queue/launcher logs.
    if PORT="$PORT" bash run.sh "${ARGS[@]}" > "$TASK_LOG" 2>&1; then
        log "DONE $CURRENT; elapsed=$((SECONDS - START))s"
    else
        RC=$?
        log "FAILED $CURRENT; exit=$RC; elapsed=$((SECONDS - START))s; log=$TASK_LOG"
        exit "$RC"
    fi
done
CURRENT=complete
log "ALL DONE: $DATASET"
