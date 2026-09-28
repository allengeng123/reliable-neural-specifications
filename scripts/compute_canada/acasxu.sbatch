#!/bin/bash
#SBATCH --job-name=neural-specs-acasxu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=acasxu-%j.out
#SBATCH --error=acasxu-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
module load python/3.11
source "${NEURAL_SPECS_VENV:-.venv}/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q
python -m neural_specs demo --output "results/slurm-${SLURM_JOB_ID}" --timeout 30
