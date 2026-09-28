# Validation and reproduction

The validation suite checks that NAP extraction follows Algorithm 1 and behaves
consistently across supported pretrained FCNs. It includes synthetic tests,
reference-model checks, and a small ACAS Xu verification example.

## Run the tests

```bash
python -m pip install -e '.[onnx,verify,test]'
python -m pytest -q
```

The 42 tests cover inference, normalization, input validation, exact threshold
boundaries, zero activations, label policies, serialization, ONNX tracing,
unequal layer widths, and SAT/UNSAT/timeout handling. Optional integrations
are skipped when their dependencies are absent; install all extras above for
the complete suite.

## Check the reference models

Download the pinned models and MNIST data, then run the checks:

```bash
python scripts/validate_extraction.py fetch
python scripts/validate_extraction.py run --output results/extraction.json
python -m neural_specs demo
```

The fetch step needs internet access. Model evaluation uses one CPU and works
offline once the assets are present. The script records source commits,
download checksums, model layouts, and runtime versions in its JSON output.
See the optional [cluster execution guide](compute-canada.md) for Slurm jobs.

## Reference results

| Model family | Models | Samples per model | Decision rule | Result |
| --- | ---: | --- | --- | --- |
| VNN-COMP 2023 ACAS Xu | 45 | 64 seeded domain samples | argmin | PASS |
| VNN-COMP 2022 MNIST FC | 3: 256×2, 256×4, 256×6 | 1,024 train + 256 held-out images | argmax | PASS |

The recorded run used Python 3.11.5, NumPy 1.26.4, ONNX 1.17.0, ONNX Runtime
1.18.0, and Marabou 2.0.0. Extraction checks took approximately 40 seconds on
one CPU. All 42 unit and integration tests passed.

For each model, the checks establish:

- The instrumented graph preserves the original graph's scores on 16 inputs.
- Pre-ReLU signs agree with post-ReLU positive states on every mining input.
- An independent ONNX interpreter agrees on every hidden neuron's activation
  state on 16 inputs.
- Patterns match an independent scalar implementation of Algorithm 1 at
  `delta = 0.8, 0.9, 0.95, 0.99, 1.0`.
- Counts and class supports are identical for batch sizes 1, 31, and 256.
  MNIST is checked with both supplied labels and model predictions.

The bundled ACAS Xu `.nnet` model and its ONNX counterpart also produce equal
activation states and extracted patterns on all 64 sampled inputs. Maximum
score difference is 2.23e-7 between their float64 and float32 evaluations.
The ACAS Xu demo verifies all four competing outputs in its stated local NAP
region after checking that the region is feasible.

Machine-readable results:

- [Extraction results for all 48 models](validation/extraction-audit.json).
- [ACAS Xu verification result](validation/acasxu-verification.json).

These checks assess extraction correctness and numerical consistency on the
stated inputs. NAP precision, recall, and formal robustness are separate
properties; a mined pattern is not automatically a robustness certificate.
The examples do not reproduce the paper's full MNIST/CIFAR-10 experiments.
