#!/bin/bash
#SBATCH --job-name=exp_rand_made
#SBATCH --output=exp-rand-%j.out
#SBATCH --error=exp-rand-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:1

source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

uv run scripts/run_baseline_experiments.py \
  --agent-configs random_generator_baseline \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --budget 50 \
  --num-episodes 3 \
  --use-wandb