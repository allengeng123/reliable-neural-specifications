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
python -m pip install 'setuptools>=68' wheel 'numpy>=1.24,<2' 'pytest>=8,<9'
python -m pip install --no-build-isolation -e .
```

The cluster wheelhouse might not contain Marabou. Download its Python 3.11 Linux
wheel from PyPI on the login node, verify the published digest, and install it:

```bash
python scripts/compute_canada/download_marabou.py
python -m pip install --no-deps --target "$VIRTUAL_ENV/marabou" \
  --platform manylinux2014_x86_64 \
  ./maraboupy-2.0.0-cp311-cp311-manylinux*.whl
export PYTHONPATH="$VIRTUAL_ENV/marabou${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from maraboupy import MarabouCore; print("Marabou native import OK")'
```

Nibi's StdEnv/2023 Python rejects the wheel's manylinux platform tag under a
normal install. The explicit platform installs only this pinned wheel into an
isolated directory, without replacing the Alliance's NumPy. The batch script
adds that directory to `PYTHONPATH` and requires the native import to succeed
before testing. This setup must be validated on the target cluster; an import
failure is not a reason to ignore failed tests. The download script is pinned
to Linux x86-64 / CPython 3.11, matching this job's module.

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

## FCN extraction audit: ACAS Xu and MNIST

Prepare the optional inference dependencies and benchmark assets on the login
node, using the same environment as above:

```bash
python -m pip install 'onnx>=1.16,<1.18' 'onnxruntime>=1.18,<1.21'
python scripts/validate_extraction.py fetch
sbatch --account=YOUR_CPU_ALLOCATION scripts/compute_canada/extraction.sbatch
```

This bounded audit requests one CPU, 2 GiB, and 15 minutes. It checks all 45
ACAS Xu models with 64 inputs each, and the three official MNIST FC models with
1,024 training and 256 held-out images. It trains no model and runs no full
robustness benchmark sweep. The script performs the synthetic tests, extraction
audit, and original ACAS Xu Marabou regression in one job. Its result is
`results/extraction-JOB_ID.json`.
