#!/bin/bash
#SBATCH --job-name=patch_chem_made
#SBATCH --output=patch-chem-%j.out
#SBATCH --error=patch-chem-%j.err
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

source .venv/bin/activate

export CUDA_HOME=/usr/project/xtmp/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python scripts/fix_chemeleon.py