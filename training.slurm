#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="ais-of-spades"
#SBATCH --time=04:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --exclude=idun-09-[02-11,15,19-20]
#SBATCH --cpus-per-task=14       
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/output_combined.txt
#SBATCH --error=slurm_outputs/output_combined.err
#SBATCH --mail-user=federih@stud.ntnu.no
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"    

uv run python src/main.py --num-workers $((SLURM_CPUS_PER_TASK - 2)) --resume-latest

echo "Job finished"