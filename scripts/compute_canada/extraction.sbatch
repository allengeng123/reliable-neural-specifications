#!/bin/bash
#SBATCH --job-name=nap-extraction
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=extraction-%j.out
#SBATCH --error=extraction-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load python/3.11
source "${NEURAL_SPECS_VENV:-.venv}/bin/activate"
if [[ -d "$VIRTUAL_ENV/marabou" ]]; then
    export PYTHONPATH="$VIRTUAL_ENV/marabou${PYTHONPATH:+:$PYTHONPATH}"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import onnx, onnxruntime; from maraboupy import MarabouCore'
python -m pytest -q
python scripts/validate_extraction.py run --output "results/extraction-${SLURM_JOB_ID}.json"
python -m neural_specs demo --output "results/acas-regression-${SLURM_JOB_ID}"
