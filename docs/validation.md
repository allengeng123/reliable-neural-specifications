# Validation

## Synthetic tests

At source commit `e91ca53`, all **23 tests passed** on Linux / Python 3.11
with Marabou 2.0.0. [Successful CI run](https://github.com/allengeng123/reliable-neural-specifications/actions/runs/36372005344).

Tests cover inference, normalization, malformed input, mining thresholds,
one-example classes, missing classes, model-bound pattern serialization,
Marabou variable mapping for unequal layer widths, SAT counterexamples,
UNSAT output queries, infeasible regions, ties, and timeout handling.

The optional TensorFlow/ONNX parser warnings emitted by Marabou are expected:
this repository uses its `.nnet` parser and does not require those packages.

## ACAS Xu

Executed via SSH on **Nibi**, on a Slurm compute node, on 2026-09-28 UTC
(2026-09-27 in America/Toronto).

| Item | Recorded value |
| --- | --- |
| Source commit | `e91ca53cea984ae982d646198a08131fdd536e93` |
| Slurm job | `22814332` |
| Node | `c255` |
| Scheduler result | `COMPLETED`, exit `0:0` |
| Allocation | 1 CPU, 2 GiB RAM, no GPU |
| Entire job elapsed | 4 seconds |
| Peak batch RSS | 89,688 KiB (about 87.6 MiB) |
| Python | 3.11.5, StdEnv/2023 |
| NumPy | 1.26.4+computecanada |
| Marabou | 2.0.0, pinned Linux CPython 3.11 wheel |
| Synthetic tests | 23 passed in 0.68 seconds |

Commands executed by the batch script:

```bash
python -m pytest -q
python -m neural_specs demo --output results/slurm-22814332 --timeout 30
```

The environment used the isolated wheel directory described in
[the cluster guide](compute-canada.md), with its published SHA-256 checked.
The runner was subsequently updated to add this directory automatically and
check the native import before running tests; the numerical code is unchanged.

### Recorded evidence

- [Full, unmodified demo report](validation/acasxu-report.json)
- [Full, unmodified mined patterns and coverage metadata](validation/acasxu-patterns.json)

The demo used 2,048 mining inputs, 512 held-out inputs, seed 2023, and
`delta=0.95`. All five predicted classes were represented. Eight inference
outputs agreed with Marabou's independent evaluator to a maximum absolute
error of **1.11e-16**.

The first held-out sample matching its own class NAP was index 3, predicted
advisory **3 (strong left)**. Its statistically mined NAP constrains **68 of
300 hidden neurons**. Verification used its normalized radius-0.001 box,
with zero activation/output margins.

| Query | Solver result | Time (seconds) |
| --- | --- | ---: |
| Input box intersected with NAP | SAT; witness independently validated | 0.1212 |
| Violation against advisory 0 | UNSAT | 0.0402 |
| Violation against advisory 1 | UNSAT | 0.0353 |
| Violation against advisory 2 | UNSAT | 0.0358 |
| Violation against advisory 4 | UNSAT | 0.0355 |

The resulting status is **`verified` for this local NAP region**. It is not
a full-domain guarantee. The small samples were labeled by the model itself;
they are a software usability check, not external ground-truth evaluation.

Global held-out empirical coverage illustrates why the local scope matters:

| Advisory | Held-out class support | NAP precision | NAP recall |
| --- | ---: | ---: | ---: |
| 0 | 432 | 1.0000 | 0.6250 |
| 1 | 14 | 0.0368 | 0.3571 |
| 2 | 22 | 0.0625 | 0.4091 |
| 3 | 26 | 0.1389 | 0.3846 |
| 4 | 18 | 0.1008 | 0.6667 |

These are empirical agreement metrics for NAPs over the whole sampled domain,
not the solver-certified local region. This demonstration does not establish
an advantage over an input-only baseline or reproduce the large paper results.
