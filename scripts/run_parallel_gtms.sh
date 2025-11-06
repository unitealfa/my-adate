#!/usr/bin/env bash
set -euo pipefail

# Default configuration for an Intel i7 11th gen CPU with 16 GB of RAM.
# You can override the seeds or parallelism with environment variables.

SEEDS=(${SEEDS_OVERRIDE:-21 22 23 24 25 26})
MAX_PROCS=${MAX_PROCS:-6}
SOLVER_CMD=${SOLVER_CMD:-"python -m gtms_cert.run_with_custom_trucks --trucks 5 --clients 200 --no-save"}
RUNS_DIR=${RUNS_DIR:-runs}

mkdir -p "${RUNS_DIR}"

pids=()
active=0

launch_solver() {
  local seed="$1"
  local log="${RUNS_DIR}/seed_${seed}.log"
  echo "[INFO] Launching solver with seed ${seed} -> ${log}"
  (
    ${SOLVER_CMD} \
      --seed "${seed}" \
      >"${log}" 2>&1
  ) &
  pids+=("$!")
}

for seed in "${SEEDS[@]}"; do
  while (( active >= MAX_PROCS )); do
    wait -n
    ((active--))
  done
  launch_solver "${seed}"
  ((active++))
  sleep 1

done

wait

echo "[INFO] All solver processes have completed. Logs stored in ${RUNS_DIR}/"
