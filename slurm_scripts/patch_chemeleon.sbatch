#!/bin/bash
#SBATCH --job-name=patch_chem_made
#SBATCH --output=slurm-%j-patch-chem.out
#SBATCH --error=slurm-%j-patch-chem.err
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

source .venv/bin/activate

python scripts/fix_chemeleon.py