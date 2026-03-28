#!/bin/bash -x
#SBATCH --account=hclimrep
#SBATCH --job-name=inference_features
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

echo "shell $SHELL"
. "$HOME/.local/bin/env"
. "$HOME/.bashrc"
echo "LMOD_ROOT $LMOD_ROOT"
# Does not get loaded automatically. Unsure why.
. $LMOD_ROOT/lmod/init/bash
# Load basic modules from software stack
module --force purge
module load Stages/2025

module load GCC/13.3.0
module load GCCcore/.13.3.0
module load CUDA/12

ml git/2.45.1
ml Python/3.12.3

ml uv/0.7.4

# Set environment variables to ensure GPU devices are used properly.
export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0

# Set rendezvous address so processes know who to talk to.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=INFO
echo "nccl_debug: $NCCL_DEBUG"

# Workaround for flipping links issue on JUWELS-BOOSTER.
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50

echo "Starting job."
date
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .venv/bin/activate

# Optional: make torch shared libraries visible for custom CUDA extensions.
export TORCH_LIB=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"

export UV_CACHE_DIR=/p/project1/weatherai/uv_cache_shared

# One job = one time shard.
# Override variables with: sbatch --export=ALL,START_DATE=...,END_DATE=...,RUN_ID_SUFFIX=... weathergen_inference_features.sbatch
FROM_RUN_ID="${FROM_RUN_ID:-ojoiudna}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-tarot_feat_cand_grad_full}"
RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-s00}"
STREAM="${STREAM:-ERA5}"
START_DATE="${START_DATE:-1979-01-01}"
END_DATE="${END_DATE:-2020-12-31}"

RUN_ID="${RUN_ID_PREFIX}_${RUN_ID_SUFFIX}"

echo "Running shard ${RUN_ID_SUFFIX}"
echo "FROM_RUN_ID=${FROM_RUN_ID}"
echo "RUN_ID=${RUN_ID}"
echo "DATE_RANGE=${START_DATE} -> ${END_DATE}"

srun --ntasks=1 --gres=gpu:1 uv run --offline --no-sync inference \
  --from_run_id="${FROM_RUN_ID}" \
  --run_id="${RUN_ID}" \
  --start_date="${START_DATE}" \
  --end_date="${END_DATE}" \
  --streams_output "${STREAM}" \
  --options \
    sampling_rate_target=1.0 \
    shuffle=False \
    with_fsdp=False \
    validate_with_ema=False \
    tarot_export_features=True \
    tarot_export_features_only=True \
    tarot_feature_mode=gradient \
    tarot_grad_proj_dim=2048 \
    tarot_grad_proj_seed=0

rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "Shard ${RUN_ID_SUFFIX} failed with exit code ${rc}"
  exit "${rc}"
fi

echo "Shard ${RUN_ID_SUFFIX} finished successfully."
