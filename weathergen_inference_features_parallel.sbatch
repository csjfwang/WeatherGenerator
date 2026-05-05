#!/bin/bash -x
#SBATCH --account=weatherai
#SBATCH --job-name=inference_features_44gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=11
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

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
echo "Tasks per Node: ${SLURM_NTASKS_PER_NODE:-4}"

source .venv/bin/activate
export TORCH_LIB=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export UV_CACHE_DIR=/e/scratch/weatherai/uv_cache_shared

set -euo pipefail

FROM_RUN_ID="${FROM_RUN_ID:-l6rog843}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-tarot_feat_cand_grad_${FROM_RUN_ID}}"
STREAM="${STREAM:-ERA5}"
TAROT_LOSS_CHANNEL_GROUP="${TAROT_LOSS_CHANNEL_GROUP:-}"
START_YEAR="${START_YEAR:-1979}"
END_YEAR_INCLUSIVE="${END_YEAR_INCLUSIVE:-2022}"
FINAL_END_DATE="${FINAL_END_DATE:-2022-12-31}"

SHARDS_PER_NODE=4
MAX_PARALLEL=$(( SLURM_JOB_NUM_NODES * SHARDS_PER_NODE ))
NUM_SHARDS=$(( END_YEAR_INCLUSIVE - START_YEAR + 1 ))
LAST_SHARD=$(( NUM_SHARDS - 1 ))

echo "FROM_RUN_ID=${FROM_RUN_ID}"
echo "RUN_ID_PREFIX=${RUN_ID_PREFIX}"
echo "YEAR SHARDS: ${START_YEAR}..${END_YEAR_INCLUSIVE} => s00..s$(printf '%02d' "$LAST_SHARD")"
echo "FINAL_END_DATE=${FINAL_END_DATE}"
echo "MAX_PARALLEL=${MAX_PARALLEL}"
echo "TAROT_LOSS_CHANNEL_GROUP=${TAROT_LOSS_CHANNEL_GROUP}"

run_one_shard() {
  local shard_idx="$1"
  local year=$(( START_YEAR + shard_idx ))
  local next_year=$(( year + 1 ))
  local suffix
  local start_date
  local end_date
  local run_id

  printf -v suffix 's%02d' "$shard_idx"
  printf -v start_date '%04d-01-01' "$year"
  if [ "$shard_idx" -eq "$LAST_SHARD" ]; then
    end_date="$FINAL_END_DATE"
  else
    printf -v end_date '%04d-01-01' "$next_year"
  fi
  run_id="${RUN_ID_PREFIX}_${suffix}"

  echo "Launching shard ${suffix}: ${start_date} -> ${end_date}"
  cmd=(
    uv run --offline --no-sync inference
      --from-run-id="$FROM_RUN_ID"
      --run-id="$run_id"
      --start-date="$start_date"
      --end-date="$end_date"
      --streams-output "$STREAM"
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
  if [ -n "$TAROT_LOSS_CHANNEL_GROUP" ]; then
    cmd+=(tarot_loss_channel_group="$TAROT_LOSS_CHANNEL_GROUP")
  fi
  srun --exclusive --exact -N1 -n1 --gres=gpu:1 --cpus-per-task="$SLURM_CPUS_PER_TASK"     "${cmd[@]}" 2>&1 | sed "s/^/[${suffix}] /" &
}

status=0
for ((wave_start=0; wave_start<NUM_SHARDS; wave_start+=MAX_PARALLEL)); do
  wave_end=$(( wave_start + MAX_PARALLEL - 1 ))
  if [ "$wave_end" -gt "$LAST_SHARD" ]; then
    wave_end="$LAST_SHARD"
  fi

  echo "Starting wave: shard ${wave_start} .. ${wave_end}"
  pids=()
  for ((shard_idx=wave_start; shard_idx<=wave_end; shard_idx++)); do
    run_one_shard "$shard_idx"
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done

done

if [ "$status" -ne 0 ]; then
  echo "At least one shard failed."
  exit "$status"
fi

echo "All yearly shards finished successfully."
