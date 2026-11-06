#!/bin/bash
# Evaluate a trained model (full-mesh rollouts of the test split) on one GPU.
# Arguments are passed to turtles-eval:
#   sbatch examples/slurm/eval.sbatch checkpoint=upt
#   sbatch examples/slurm/eval.sbatch model=persistence
#
# Edit the lines marked EDIT for your cluster.
#SBATCH --job-name=turtles-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=EDIT
##SBATCH --partition=EDIT
set -euo pipefail

# EDIT: load your modules and activate the environment where turtles is installed.
# module load cuda
# source ~/venvs/turtles/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
turtles-eval "$@"
