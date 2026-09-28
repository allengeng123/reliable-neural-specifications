# Run on Compute Canada / the Digital Research Alliance

Use your existing SSH configuration, for example `ssh nibi`. Complete Duo MFA
interactively. No SSH keys, account identifiers, or authentication data belong
in this repository.

On the login node, prepare an environment and clone into scratch:

```bash
cd "$SCRATCH"
git clone https://github.com/allengeng123/reliable-neural-specifications.git
cd reliable-neural-specifications
module load python/3.11
python -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy>=1.24,<2' 'pytest>=8,<9'
python -m pip install -e .
```

The cluster wheelhouse might not contain Marabou. Download its Python 3.11 Linux
wheel from PyPI on the login node, verify the published digest, and install it:

```bash
python scripts/compute_canada/download_marabou.py
python -m pip install ./maraboupy-2.0.0-cp311-cp311-manylinux*.whl
```

Submit from the repository root with your CPU allocation:

```bash
sbatch --account=YOUR_CPU_ALLOCATION scripts/compute_canada/acasxu.sbatch
```

The job requests one CPU, 2 GiB RAM, and ten minutes. It runs the synthetic test
suite followed by the small ACAS Xu demo. Each solver query is bounded to 30
seconds. No GPU is requested. Installation is done before submission so the
compute job needs no network access.

Inspect the returned job ID:

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
cat acasxu-JOB_ID.out
cat results/slurm-JOB_ID/report.json
```

A nonzero job status must be investigated. A verification exit code of 2 means
the report was written but the property was not certified. It may indicate a
counterexample, an empty region, or an inconclusive query; read the JSON status.
`NEURAL_SPECS_VENV` can point to an existing environment if needed.
