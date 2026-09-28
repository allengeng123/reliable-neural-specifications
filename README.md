# Towards Reliable Neural Specifications

**This is the reorganized code repository for the ICML 2023 Oral paper
[Towards Reliable Neural Specifications](https://icml.cc/virtual/2023/oral/25466).**

Chuqin Geng, Nham Le, Xiaojie Xu, Zhaoyue Wang, Arie Gurfinkel, and Xujie Si.

[ICML oral page](https://icml.cc/virtual/2023/oral/25466) ·
[Paper and citation](https://proceedings.mlr.press/v202/geng23a.html) ·
[PDF](https://proceedings.mlr.press/v202/geng23a/geng23a.pdf)

This focused edition reorganizes the ACAS Xu workflow from
[VerifyNNE](https://github.com/allengeng123/VerifyNNE) and the related verification
work in [Verify-Network](https://github.com/allengeng123/Verify-Network).
It provides reusable extraction of **neural activation patterns (NAPs)** from
pretrained **ReLU fully connected networks**, plus a small ACAS Xu
demonstration checking output guarantees with **Marabou**.

**Scope:** generic `.nnet` / ONNX ReLU FCN extraction, with ACAS Xu and the
official VNN-COMP MNIST FC models as validation cases. Formal verification in
this edition remains ACAS Xu focused. This does **not** reproduce the paper's
large MNIST/CIFAR-10 tables, train models, or claim full aircraft-system safety.
See [extraction guide](docs/extraction.md) and [migration notes](docs/migration.md).

## Quick start

Use **Python 3.11 on Linux** for the complete workflow. The pinned Marabou 2.0.0
wheel also supports Python 3.10 on Linux. Core inference and pattern mining work
without Marabou, including on Windows.

```bash
git clone https://github.com/allengeng123/reliable-neural-specifications.git
cd reliable-neural-specifications
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[verify,test]'
```

Run the bounded demonstration (one bundled network, one CPU, no GPU):

```bash
python -m neural_specs demo
```

For **Compute Canada / Digital Research Alliance**, use
[the SSH and Slurm instructions](docs/compute-canada.md). Submit the experiment
to a compute node instead of running it on the login node.

The demo:

1. Loads the bundled `ACASXU_run2a_1_1_batch_2000.nnet` network in float64.
2. Mines per-class NAPs from 2,048 seeded uniform samples and measures empirical
   precision/recall on 512 separate samples. Labels are the model's **argmin**
   predictions, not external ground truth.
3. Cross-checks eight inference results against Marabou's independent evaluator.
4. Selects the first held-out input matching its predicted class's mined NAP.
5. Checks that its NAP region inside a normalized radius-0.001 box is nonempty,
   then checks strict output dominance against all four competing advisories.

The local box makes this a small usability demonstration. It is explicitly
recorded in the report; a local result is **not** a full-domain NAP certificate.
Each solver query has a 30-second timeout. The default demo performs five
queries: one feasibility query and four output queries.

Outputs are `results/demo/patterns.json` and `results/demo/report.json`.
Reports record model SHA-256, seeds, dependency versions, sample counts, domain
bounds, empirical coverage, solver statuses, elapsed times, and SAT witnesses.

**Validated on Nibi:** 42 tests passed, extraction passed on all **45 ACAS Xu
and three MNIST FC networks**, and the ACAS Xu Marabou regression still
verified all four competing outputs. The complete one-CPU audit job took
49 seconds. [Recorded runs and full JSON evidence](docs/validation.md).

## Separate mining and verification

Only NumPy is needed for mining:

```bash
python -m pip install -e .
python -m neural_specs mine --samples 2048 --holdout 512 --delta 0.95 --seed 2023
```

Optionally verify one mined class over the **full** normalized `.nnet` domain:

```bash
python -m neural_specs verify \
  --patterns results/mine/patterns.json --label 0 --timeout 30
```

Full-domain queries can produce counterexamples or time out. Neither means the
program failed, and neither is reported as verified. `--model path/to/model.nnet`
supports another ACAS Xu network without editing source files. Run commands
from the repository root or pass an explicit model path.

## Extract from another pretrained FCN

The `extract` command accepts arbitrary hidden-layer depths and widths, keeping
neuron identities from the actual model. Provide an NPZ dataset with `inputs`
and, for dataset-label grouping, integer `labels`:

```bash
python -m pip install -e '.[onnx]'
python -m neural_specs extract \
  --model path/to/pretrained.onnx --data path/to/data.npz \
  --label-source provided --decision argmax --delta 0.95
```

Use `--label-source predicted --decision argmin` for ACAS Xu model predictions;
use `argmax` for MNIST. The choice is explicit and never inferred from a model
filename. `.nnet`, `.onnx`, and `.onnx.gz` are supported. See the
[guide](docs/extraction.md) for MNIST commands, preprocessing, the streaming
Python API, supported graph formats, and the exact Algorithm 1 conventions.

The official MNIST FC benchmark is in **VNN-COMP 2022**, with 256×2, 256×4,
and 256×6 networks. The official 2023 repository contains ACAS Xu but no
`mnist_fc` category; benchmark years are recorded separately in the audit.

## Verification outcomes

| Result | Meaning |
| --- | --- |
| `verified` | Region is feasible; all competing-output queries are UNSAT |
| `counterexample` | A SAT witness violates strict dominance, independently re-evaluated |
| `empty_region` | The input bounds and NAP have no satisfying input |
| `inconclusive` | A timeout, unknown result, or unvalidated witness prevents a conclusion |

Exit codes: `0` for successful extraction/mining or verified output, `2` for other completed
verification outcomes, and `1` for invalid input/dependency/runtime failures.
ACAS Xu chooses the **lowest** output; an output tie counts as a violation of
the strict dominance property. See [method and numerical conventions](docs/method.md).

## Layout

```text
src/neural_specs/
  nnet.py             .nnet loading, normalization, inference, activation traces
  patterns.py         statistical mining, coverage, validated JSON interchange
  extraction.py       streaming model-independent extraction and label policy
  onnx_fcn.py         trace actual ONNX ReLU sites with ONNX Runtime
  verification.py     Marabou queries and independent SAT-witness checks
  cli.py              extract / mine / verify / demo commands
data/acasxu/           one small model, checksum, and attribution
scripts/compute_canada/acasxu.sbatch
scripts/compute_canada/extraction.sbatch
scripts/validate_extraction.py  pinned benchmark downloads and extraction audit
tests/                synthetic unit and solver integration tests
docs/                 method, migration, and cluster instructions
```

```bash
python -m pytest -q                 # includes solver tests when Marabou is installed
python -m pytest -q -m 'not solver' # core-only checks
```

CI runs synthetic tests. ACAS Xu experiment evidence, when available, is recorded
in [validation notes](docs/validation.md).

## Citation

```bibtex
@InProceedings{pmlr-v202-geng23a,
  title = {Towards Reliable Neural Specifications},
  author = {Geng, Chuqin and Le, Nham and Xu, Xiaojie and Wang, Zhaoyue
            and Gurfinkel, Arie and Si, Xujie},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {11196--11212},
  year = {2023},
  volume = {202},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v202/geng23a.html}
}
```

## Attribution

See [NOTICE](NOTICE.md) for the historical source commits, third-party model
attribution, and licensing status. The bundled model retains its CC BY 4.0
notice. This cleanup does not relicense either historical repository.
