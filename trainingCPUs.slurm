#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="ais-of-spades-discussed"
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --mail-user=federih@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

JOB_NAME="${JOB_NAME:-${SLURM_JOB_NAME}}"
mkdir -p "slurm_outputs/${JOB_NAME}"
exec > "slurm_outputs/${JOB_NAME}/output.txt" 2> "slurm_outputs/${JOB_NAME}/output.err"

scontrol update JobId="${SLURM_JOB_ID}" JobName="${JOB_NAME}" 2>/dev/null || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running from this directory: ${SLURM_SUBMIT_DIR}"
echo "Name of job: ${JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"

uv run python src/main.py --num-workers $((SLURM_CPUS_PER_TASK - 1)) --resume-latest --run-name "${JOB_NAME}"

echo "Job finished"

