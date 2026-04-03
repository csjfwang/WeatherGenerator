#!/bin/bash -x
#SBATCH --account=hclimrep
#SBATCH --job-name=select_tarot_indices
#SBATCH --time=24:00:00
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

CANDIDATE_RUN_ID="${CANDIDATE_RUN_ID:?Please pass CANDIDATE_RUN_ID, e.g. sbatch --export=ALL,CANDIDATE_RUN_ID=l6rog843,TARGET_RUN_ID_PREFIX=tarot_feat_tgt_grad_l6rog843,OUTPUT_TAG=tarot_2022_l6rog843 weathergen_select_tarot_indices.sbatch}"
TARGET_RUN_ID_PREFIX="${TARGET_RUN_ID_PREFIX:?Please pass TARGET_RUN_ID_PREFIX, e.g. tarot_feat_tgt_grad_l6rog843 or tarot_feat_tgt_grad_2023_l6rog843}"
OUTPUT_TAG="${OUTPUT_TAG:?Please pass OUTPUT_TAG, e.g. tarot_2022_l6rog843 or tarot_2023_l6rog843}"
CANDIDATE_GLOB="/p/scratch/weatherai/shared_work/results/tarot_feat_cand_grad_${CANDIDATE_RUN_ID}_s*/tarot_features_chkpt*_rank*.npz"
TARGET_GLOB="/p/scratch/weatherai/shared_work/results/${TARGET_RUN_ID_PREFIX}_s*/tarot_features_chkpt*_rank*.npz"
SELECTED_IDS_OUT="TAROT/${OUTPUT_TAG}_selected_ids_train_full.npy"
SELECTED_GLOBAL_IDX_OUT="TAROT/${OUTPUT_TAG}_selected_indices_train_full.npy"

echo "CANDIDATE_RUN_ID=${CANDIDATE_RUN_ID}"
echo "TARGET_RUN_ID_PREFIX=${TARGET_RUN_ID_PREFIX}"
echo "OUTPUT_TAG=${OUTPUT_TAG}"
echo "CANDIDATE_GLOB=${CANDIDATE_GLOB}"
echo "TARGET_GLOB=${TARGET_GLOB}"
echo "SELECTED_IDS_OUT=${SELECTED_IDS_OUT}"
echo "SELECTED_GLOBAL_IDX_OUT=${SELECTED_GLOBAL_IDX_OUT}"

srun --ntasks=1 uv run --offline --no-sync python scripts/select_tarot_indices.py   --candidate "${CANDIDATE_GLOB}"   --target "${TARGET_GLOB}"   --feature grad_projected   --method fixed_size   --ratio 0.2   --dedup-policy mean   --score-block-size 4096   --output "${SELECTED_IDS_OUT}"   --output-global-idx "${SELECTED_GLOBAL_IDX_OUT}"   --global-index-origin 1979-01-01   --global-index-step-hours 6   --save-score-analysis

rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "TAROT index selection failed with exit code ${rc}"
  exit "${rc}"
fi

echo "TAROT index selection finished successfully."
