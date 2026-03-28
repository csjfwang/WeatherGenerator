#!/bin/bash -x
#SBATCH --account=hclimrep
#SBATCH --job-name=select_tarot_indices
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
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
# Load basic modules from software stack.
module --force purge
module load Stages/2025

module load GCC/13.3.0
module load GCCcore/.13.3.0
module load CUDA/12

ml git/2.45.1
ml Python/3.12.3
ml uv/0.7.4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export UV_CACHE_DIR=/p/project1/weatherai/uv_cache_shared

source .venv/bin/activate

echo "Starting TAROT index selection job."
date
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"

srun --ntasks=1 uv run --offline --no-sync python scripts/select_tarot_indices.py \
  --candidate "/p/scratch/weatherai/shared_work/results/tarot_feat_cand_grad_full_s*/tarot_features_chkpt*_rank*.npz" \
  --target "/p/scratch/weatherai/shared_work/results/tarot_feat_tgt_grad_full_s*/tarot_features_chkpt*_rank*.npz" \
  --feature grad_projected \
  --method fixed_size \
  --ratio 0.2 \
  --dedup-policy mean \
  --score-block-size 4096 \
  --output TAROT/selected_ids_train_full.npy \
  --output-global-idx TAROT/selected_indices_train_full.npy \
  --global-index-origin 1979-01-01 \
  --global-index-step-hours 6 \
  --candidate-shard-start-dates "1979-01-01,1984-04-01,1989-07-01,1994-10-01,2000-01-01,2005-04-01,2010-07-01,2015-10-01"

rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "TAROT index selection failed with exit code ${rc}"
  exit "${rc}"
fi

echo "TAROT index selection finished successfully."
