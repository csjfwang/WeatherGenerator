#!/bin/bash -x
#SBATCH --account=e-ext-2025e01-128
#SBATCH --job-name=train
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

echo "shell $SHELL"
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi

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

# set some environment variables to ensure that GPU-devices are used properly
export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
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

# work-around for flipping links issue on JUWELS-BOOSTER
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

export UV_CACHE_DIR=/e/scratch/weatherai/uv_cache_shared

# srun uv run --offline --no-sync train_continue --from_run_id r93m2nvw
# srun uv run --offline --no-sync train_continue \
#   --from-run-id l6rog843 \
#   --mini-epoch -1 \
#   --run-id l6rog843_ft_tarot_2022Q4_2023Q3 \
#   --config config/config_forecasting_finetuning.yml config/tarot_2022Q4_2023Q3_l6rog843.yml 

# srun uv run --offline --no-sync train_continue \
#   --from-run-id epiys5z8_ft_tarot_2022Q4_2023Q3_smlr \
#   --mini-epoch -1 \
#   --reuse-run-id 
#   --config config/config_forecasting_finetuning.yml config/tarot_2022Q4_2023Q3_epiys5z8.yml 
  # --options istep=0 data_loader_rng_seed=42

# srun uv run --offline --no-sync train_continue \
#   --from-run-id epiys5z8_ft_tarot_2022Q4_2023Q3 \
#   --mini-epoch -1 \
#   --reuse-run-id 
#   --config config/config_forecasting_finetuning.yml config/tarot_2022Q4_2023Q3_epiys5z8.yml 
  
# srun uv run --offline --no-sync train_continue \
#   --from-run-id he7ut6vx \
#   --mini-epoch -1 \
#   --run-id he7ut6vx_ft_tarot_2022Q4_2023Q3 \
#   --config config/config_forecasting_finetuning_matz.yml config/tarot_2022Q4_2023Q3_epiys5z8.yml 

srun uv run --offline --no-sync train_continue \
  --from-run-id he7ut6vx \
  --mini-epoch -1 \
  --run-id he7ut6vx_ft_tarot_top9year \
  --config config/config_forecasting_finetuning_matz.yml config/top9years_2022_2017_2014_2011_2009.yml

# srun uv --offline run train

# srun uv run --offline --no-sync train --config config/tarot_20pct.yml
# srun uv run --offline --no-sync train --config config/recent_20pct.yml
# srun uv run --offline --no-sync train --config config/full_100pct.yml
# srun uv run --offline --no-sync train --config config/stratified_20pct.yml
# srun uv run --offline --no-sync train --config config/random_20pct.yml

# Full data 32 epochs + 20% data 32 epochs

# srun uv run --offline --no-sync train_continue \
#   --from_run_id ojoiudna \
#   --mini_epoch -1 \
#   --run_id full2tarot20 \
#   --config config/tarot_20pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42


# srun uv run --offline --no-sync train_continue \
#   --from_run_id ojoiudna \
#   --mini_epoch -1 \
#   --run_id full2random20 \
#   --config config/random_20pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42

# srun uv run --offline --no-sync train_continue \
#   --from_run_id ojoiudna \
#   --mini_epoch -1 \
#   --run_id full2recent20 \
#   --config config/recent_20pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42

# srun uv run --offline --no-sync train_continue \
#   --from_run_id ojoiudna \
#   --mini_epoch -1 \
#   --run_id full2stratified20 \
#   --config config/stratified_20pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42


# srun uv run --offline --no-sync train_continue \
#   --from_run_id ojoiudna \
#   --mini_epoch -1 \
#   --run_id full2full \
#   --config config/full_100pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42



# full data 64 epochs (run_id: r0pn5rws) + 20% finetune 32 epochs

# srun uv run --offline --no-sync train_continue \
#   --from_run_id r0pn5rws \
#   --mini_epoch -1 \
#   --run_id 64full2tarot20 \
#   --config config/tarot_20pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42


# srun uv run --offline --no-sync train_continue \
#   --from_run_id r0pn5rws \
#   --mini_epoch -1 \
#   --run_id 64full2random20 \
#   --config config/random_20pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42

# srun uv run --offline --no-sync train_continue \
#   --from_run_id r0pn5rws \
#   --mini_epoch -1 \
#   --run_id 64full2recent20 \
#   --config config/recent_20pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42

# srun uv run --offline --no-sync train_continue \
#   --from_run_id r0pn5rws \
#   --mini_epoch -1 \
#   --run_id 64full2stratified20 \
#   --config config/stratified_20pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42


# srun uv run --offline --no-sync train_continue \
#   --from_run_id r0pn5rws \
#   --mini_epoch -1 \
#   --run_id 64full2full \
#   --config config/full_100pct.yml \
#   --options istep=0 num_mini_epochs=32 data_loader_rng_seed=42

