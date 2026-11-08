#!/bin/bash
# Train a model on a SLURM cluster. Arguments are passed to turtles-train:
#   sbatch examples/slurm/train.sbatch experiment=paper/upt run_name=upt
# Several GPUs: set --ntasks-per-node and --gpus-per-node to the same number (one process
# per GPU); the trainer reads the layout from srun.
#
# Edit the lines marked EDIT for your cluster.
#SBATCH --job-name=turtles-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=EDIT
##SBATCH --partition=EDIT
set -euo pipefail

# EDIT: load your modules and activate the environment where turtles is installed.
# module load cuda
# source ~/venvs/turtles/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

srun turtles-train "$@"   # resume an interrupted run with: run_name=<name> resume=true
