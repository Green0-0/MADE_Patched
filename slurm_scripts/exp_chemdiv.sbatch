#!/bin/bash
#SBATCH --job-name=exp_chem_mlip_made
#SBATCH --output=chem-mlip-%A_%a.out
#SBATCH --error=chem-mlip-%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=1-10

source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

uv run scripts/run_baseline_experiments.py \
  --agent-configs chemeleon_diversity_planner_comp \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --system-index $((SLURM_ARRAY_TASK_ID - 1)) \
  --budget 50 \
  --num-episodes 5 \
  --use-wandb
