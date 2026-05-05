#!/bin/bash -x
#SBATCH --account=weatherai
#SBATCH --job-name=inference_features_4gpu
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

set -o pipefail

echo "shell $SHELL"
. "$HOME/.local/bin/env"
. "$HOME/.bashrc"
echo "LMOD_ROOT $LMOD_ROOT"
. "$LMOD_ROOT/lmod/init/bash"
module --force purge
module load Stages/2025
module load GCC/13.3.0
module load GCCcore/.13.3.0
module load CUDA/12
ml git/2.45.1
ml Python/3.12.3
ml uv/0.7.4

export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    MASTER_ADDR="${MASTER_ADDR}i"
fi
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting job."
date
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"

source .venv/bin/activate
export TORCH_LIB=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export UV_CACHE_DIR=/e/scratch/weatherai/uv_cache_shared

FROM_RUN_ID="${FROM_RUN_ID:-l6rog843}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-tarot_feat_tgt_grad_${FROM_RUN_ID}}"
STREAM="${STREAM:-ERA5}"
TAROT_LOSS_CHANNEL_GROUP="${TAROT_LOSS_CHANNEL_GROUP:-}"
export TARGET_START_DATE="${TARGET_START_DATE:-2022-01-01}"
export TARGET_END_DATE="${TARGET_END_DATE:-2023-01-01}"

# Split the TAROT target interval into four quarterly sub-intervals.
suffixes=(s00 s01 s02 s03)
readarray -t quarter_bounds < <(
  python3 - <<'PY2'
from datetime import date
import os

start = date.fromisoformat(os.environ["TARGET_START_DATE"])
end = date.fromisoformat(os.environ["TARGET_END_DATE"])

bounds = [start]
cur = start
for _ in range(4):
    month = cur.month + 3
    year = cur.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    cur = date(year, month, 1)
    bounds.append(cur)

if bounds[-1] != end:
    raise SystemExit(
        f"TARGET_START_DATE={start.isoformat()} and TARGET_END_DATE={end.isoformat()} must span exactly four quarterly sub-intervals. Expected end={bounds[-1].isoformat()}."
    )

for bound in bounds:
    print(bound.isoformat())
PY2
)
starts=("${quarter_bounds[0]}" "${quarter_bounds[1]}" "${quarter_bounds[2]}" "${quarter_bounds[3]}")
ends=("${quarter_bounds[1]}" "${quarter_bounds[2]}" "${quarter_bounds[3]}" "${quarter_bounds[4]}")

shard_indices=(0 1 2 3)

echo "Will run shard indices: ${shard_indices[*]}"
echo "TARGET_START_DATE=${TARGET_START_DATE}"
echo "TARGET_END_DATE=${TARGET_END_DATE}"
echo "TAROT_LOSS_CHANNEL_GROUP=${TAROT_LOSS_CHANNEL_GROUP}"


run_one_shard() {
  local idx="$1"
  local suffix="${suffixes[$idx]}"
  local start_date="${starts[$idx]}"
  local end_date="${ends[$idx]}"
  local run_id="${RUN_ID_PREFIX}_${suffix}"

  echo "Launching shard ${suffix}: ${start_date} -> ${end_date}"
  srun --exclusive --exact -N1 -n1 --gres=gpu:1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash -c '
      set -euo pipefail
      if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "[ERROR] CUDA_VISIBLE_DEVICES is empty for shard '"${suffix}"'" >&2
        exit 3
      fi
      echo "Shard '"${suffix}"' on host $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      nvidia-smi -L
      cmd=(
        uv run --offline --no-sync inference
          --from-run-id="'"${FROM_RUN_ID}"'"
          --run-id="'"${run_id}"'"
          --start-date="'"${start_date}"'"
          --end-date="'"${end_date}"'"
          --streams-output "'"${STREAM}"'"
          --options
            sampling_rate_target=1.0
            shuffle=False
            with_fsdp=False
            validate_with_ema=False
            tarot_export_features=True
            tarot_export_features_only=True
            tarot_feature_mode=gradient
            tarot_grad_proj_dim=2048
            tarot_grad_proj_seed=0
            tarot_grad_wrt=".*forecast_engine.*" 
            train_logging.terminal=10 
            train_logging.metrics=20 
            train_logging.checkpoint=250 
            train_logging.log_grad_norms=False
      )
      if [ -n "'"${TAROT_LOSS_CHANNEL_GROUP}"'" ]; then
        cmd+=(tarot_loss_channel_group="'"${TAROT_LOSS_CHANNEL_GROUP}"'")
      fi
      "${cmd[@]}"
    ' 2>&1 | sed "s/^/[${suffix}] /" &
}

pids=()
for shard_idx in "${shard_indices[@]}"; do
  run_one_shard "$shard_idx"
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo "At least one shard failed."
  exit "$status"
fi

echo "All four target shards finished successfully."

