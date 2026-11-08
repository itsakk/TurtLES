#!/bin/bash
# Run one simulation stage of a case created by `turtles-gen setup`:
#   sbatch examples/slurm/generate_stage.sbatch transient  <cases_dir>/<case>
#   sbatch examples/slurm/generate_stage.sbatch extend     <cases_dir>/<case>   # while not stationary
#   sbatch examples/slurm/generate_stage.sbatch trajectory <cases_dir>/<case>
# Then convert the snapshots (CPU only): turtles-gen run convert <cases_dir>/<case>
#
# One MPI rank per GPU: keep --ntasks-per-node and --gpus-per-node equal to `gpus` in
# the pipeline settings (4 for TurtLES). Edit the lines marked EDIT for your cluster.
#SBATCH --job-name=turtles-gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=EDIT
##SBATCH --partition=EDIT
set -euo pipefail

# EDIT: load MPI/CUDA and activate the environment with PyFR 2.0.3 and turtles[generation].
# module load openmpi cuda
# source ~/venvs/pyfr/bin/activate

STAGE=$1
CASE=$2
# EDIT if needed: some MPI builds need "srun --mpi=pmi2" or "srun --mpi=pmix".
turtles-gen run "$STAGE" "$CASE" --launcher "srun"
