# Towards Reliable Neural Specifications

**Code for the ICML 2023 Oral paper
[Towards Reliable Neural Specifications](https://icml.cc/virtual/2023/oral/25466).**

Chuqin Geng, Nham Le, Xiaojie Xu, Zhaoyue Wang, Arie Gurfinkel, and Xujie Si.

[ICML oral](https://icml.cc/virtual/2023/oral/25466) ·
[Paper](https://proceedings.mlr.press/v202/geng23a/geng23a.pdf) ·
[Citation](https://proceedings.mlr.press/v202/geng23a.html)

Extract **neural activation patterns (NAPs)** from pretrained ReLU fully
connected networks and use them as specifications for neural network
verification. The package supports `.nnet` and ONNX models, with runnable
examples for **ACAS Xu** and **MNIST FCNs**.

## Setup

Use **Python 3.11 on Linux** for all features. A CPU is sufficient.

```bash
git clone https://github.com/allengeng123/reliable-neural-specifications.git
cd reliable-neural-specifications
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[onnx,verify,test]'
```

For a smaller installation, `pip install -e .` provides NumPy-based `.nnet`
inference and NAP mining. Add `[onnx]` for ONNX extraction, `[verify]` for
Marabou verification, and `[test]` for the test suite. Core inference and mining
also work on Windows; the verification examples use the Linux Marabou wheel.

## Quick start: ACAS Xu

Run the bundled pretrained network through mining and verification:

```bash
python -m neural_specs demo
```

The demo samples 2,048 inputs, mines class-specific NAPs, and evaluates their
coverage on 512 held-out inputs. It then checks whether a mined NAP guarantees
one advisory within a small input region, using Marabou to compare all four
competing outputs. No training or model download is needed.

Results are saved to:

- `results/demo/patterns.json` — selected neurons, activation states, and coverage.
- `results/demo/report.json` — input region, solver outcomes, and witnesses.

The default region is a normalized radius-0.001 box intersected with the NAP.
A successful result certifies that region; each solver query has a 30-second
timeout.

## How it works

A NAP describes which hidden neurons should be active or inactive. Inputs
sharing a pattern can be grouped even when they are far apart in input space.

1. **Collect activations.** Evaluate a pretrained model on labeled inputs and
   record each hidden neuron's pre-ReLU value. Positive values are active;
   zero and negative values are inactive.
2. **Mine a pattern for each class.** Let `p` be a neuron's activation frequency.
   Select it as active when `p >= delta`, or inactive when `p <= 1 - delta`.
   Leave other neurons unconstrained. The default is `delta = 0.95`.
3. **Evaluate or verify the pattern.** Measure empirical coverage on held-out
   inputs, or ask a solver whether any input satisfying the pattern violates
   the desired output property.

Mining follows Algorithm 1 of the paper. The threshold controls neuron
selection; joint pattern recall is measured separately. Extraction supports
arbitrary depths and widths of supported ReLU FCNs. See the
[extraction guide](docs/extraction.md) for model formats and the streaming API.

## Extract NAPs from your own model

Prepare an NPZ file with an `inputs` array and an integer `labels` array, then run:

```bash
python -m neural_specs extract \
  --model path/to/model.onnx --data path/to/data.npz \
  --label-source provided --decision argmax --delta 0.95
```

The model can be `.nnet`, `.onnx`, or `.onnx.gz`. Supply inputs with the same
preprocessing used by the pretrained model. Outputs are
`results/extract/patterns.json` and `results/extract/counts.npz`.

Use `--label-source provided` to group by dataset labels, or `predicted` to
group by the model's decisions. **MNIST uses `argmax`; ACAS Xu uses `argmin`.**
These choices are explicit so extraction uses the intended class convention.

## Example: MNIST fully connected networks

Download the reference models and prepare small training/test datasets:

```bash
python scripts/validate_extraction.py fetch
python -m neural_specs extract \
  --model benchmarks/cache/mnist-net_256x4.onnx.gz \
  --data benchmarks/cache/mnist-train-1024.npz \
  --label-source provided --decision argmax \
  --delta 0.95 --output results/mnist
```

The example uses float32 pixels scaled to `[0, 1]`. Replace `256x4` with
`256x2` or `256x6` to use the other official VNN-COMP 2022 MNIST FC models.
The fetch command also downloads the ACAS Xu reference models used by the
validation suite. Downloaded assets stay in `benchmarks/cache/`.

## Verify an ACAS Xu pattern

Mine patterns and check one class over the model's full input domain:

```bash
python -m neural_specs mine --delta 0.95 --seed 2023
python -m neural_specs verify \
  --patterns results/mine/patterns.json --label 0 --timeout 30
```

Use `--model path/to/network.nnet` on both commands for another ACAS Xu model.
Verification checks that the target advisory has a strictly lower score than
every competitor. An output tie counts as a violation.

| Result | Meaning |
| --- | --- |
| `verified` | The region is feasible and every competing-output query is UNSAT |
| `counterexample` | A validated input violates the output property |
| `empty_region` | No input satisfies the bounds and pattern |
| `inconclusive` | A timeout or unresolved query prevents a conclusion |

Full-domain verification can return counterexamples or time out. Extraction
supports both ACAS Xu and MNIST FCNs; the supplied formal verification workflow
targets ACAS Xu. See [verification semantics](docs/method.md) for numerical
conventions and margins.

## Tests and reference results

```bash
python -m pytest -q
```

The validation suite covers 45 ACAS Xu networks and three MNIST FCNs, checking
activation traces, Algorithm 1 results, and consistency across batch sizes.
All 42 tests and the 48-model extraction checks pass in the recorded reference
run. See [validation and reproduction](docs/validation.md).

## Project structure

```text
src/neural_specs/     Model loading, activation tracing, mining, and verification
data/acasxu/         Bundled ACAS Xu model
scripts/             Benchmark preparation and validation
tests/               Unit and integration tests
docs/                Extraction, verification, and reproduction guides
```

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

Model attribution and licensing information are in [NOTICE](NOTICE.md).
